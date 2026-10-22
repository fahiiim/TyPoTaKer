using System;
using System.Runtime.InteropServices;
using System.Threading;
using System.Windows.Automation;

namespace TyPoTaKer.Desktop.Services;

public static class SelectionHelper
{
    /// <summary>
    /// Attempts to retrieve currently selected text from the active foreground control.
    /// Priority 1: UI Automation TextPattern (instant, zero side effects).
    /// Priority 2: Clipboard copy simulation (restoring previous clipboard text afterwards).
    /// </summary>
    public static string? GetSelectedText(int timeoutMs = 80)
    {
        // 1. Try UI Automation TextPattern first
        string? uiaText = TryGetSelectionFromUIA();
        if (!string.IsNullOrWhiteSpace(uiaText))
        {
            return uiaText.Trim();
        }

        // 2. Try Clipboard Copy with state restoration in dedicated STA thread
        string? clipText = TryGetSelectionFromClipboard(timeoutMs);
        if (!string.IsNullOrWhiteSpace(clipText))
        {
            return clipText.Trim();
        }

        return null;
    }

    private static string? TryGetSelectionFromUIA()
    {
        try
        {
            var focused = AutomationElement.FocusedElement;
            if (focused != null &&
                focused.TryGetCurrentPattern(TextPattern.Pattern, out object patternObj) &&
                patternObj is TextPattern textPattern)
            {
                var selection = textPattern.GetSelection();
                if (selection != null && selection.Length > 0)
                {
                    string text = selection[0].GetText(-1);
                    if (!string.IsNullOrWhiteSpace(text))
                    {
                        return text;
                    }
                }
            }
        }
        catch
        {
            // Ignore COM exceptions from switching windows
        }
        return null;
    }

    private static string? TryGetSelectionFromClipboard(int timeoutMs)
    {
        string? result = null;

        var thread = new Thread(() =>
        {
            string? originalText = null;
            try
            {
                if (System.Windows.Clipboard.ContainsText())
                    originalText = System.Windows.Clipboard.GetText();
            }
            catch { }

            try
            {
                try { System.Windows.Clipboard.Clear(); } catch { }

                KeyboardHook.IsSimulating = true;
                SendCtrlC();

                int elapsed = 0;
                while (elapsed < timeoutMs)
                {
                    Thread.Sleep(15);
                    elapsed += 15;
                    try
                    {
                        if (System.Windows.Clipboard.ContainsText())
                        {
                            result = System.Windows.Clipboard.GetText();
                            break;
                        }
                    }
                    catch { }
                }
            }
            finally
            {
                KeyboardHook.IsSimulating = false;

                if (originalText != null)
                {
                    try
                    {
                        System.Windows.Clipboard.SetText(originalText);
                    }
                    catch { }
                }
            }
        });

        thread.SetApartmentState(ApartmentState.STA);
        thread.Start();
        thread.Join(timeoutMs + 100);

        return result;
    }

    private static void SendCtrlC()
    {
        var inputs = new Win32Native.INPUT[4];

        // Ctrl down
        inputs[0] = new Win32Native.INPUT
        {
            type = Win32Native.INPUT_KEYBOARD,
            ki = new Win32Native.KEYBDINPUT { wVk = (ushort)Win32Native.VK_CONTROL, dwFlags = 0 }
        };
        // 'C' down
        inputs[1] = new Win32Native.INPUT
        {
            type = Win32Native.INPUT_KEYBOARD,
            ki = new Win32Native.KEYBDINPUT { wVk = 0x43, dwFlags = 0 } // 'C'
        };
        // 'C' up
        inputs[2] = new Win32Native.INPUT
        {
            type = Win32Native.INPUT_KEYBOARD,
            ki = new Win32Native.KEYBDINPUT { wVk = 0x43, dwFlags = Win32Native.KEYEVENTF_KEYUP }
        };
        // Ctrl up
        inputs[3] = new Win32Native.INPUT
        {
            type = Win32Native.INPUT_KEYBOARD,
            ki = new Win32Native.KEYBDINPUT { wVk = (ushort)Win32Native.VK_CONTROL, dwFlags = Win32Native.KEYEVENTF_KEYUP }
        };

        Win32Native.SendInput((uint)inputs.Length, inputs, Marshal.SizeOf<Win32Native.INPUT>());
    }
}
