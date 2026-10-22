using System;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Automation;
using System.Windows.Automation.Text;
using Point = System.Windows.Point;

namespace TyPoTaKer.Desktop.Services;

public sealed class CaretInfo
{
    public bool IsVisible { get; init; }
    public Point ScreenPosition { get; init; }
    public double CaretHeight { get; init; } = 20;
    public string? SurroundingText { get; init; }
    public bool IsPassword { get; init; }
}

public static class CaretTracker
{
    public static CaretInfo GetCaretInfo()
    {
        try
        {
            var focused = AutomationElement.FocusedElement;
            if (focused != null)
            {
                // 1. Password field safety check
                object isPasswordObj = focused.GetCurrentPropertyValue(AutomationElement.IsPasswordProperty, true);
                if (isPasswordObj is bool isPassword && isPassword)
                {
                    return new CaretInfo
                    {
                        IsVisible = false,
                        IsPassword = true,
                        ScreenPosition = new Point(0, 0)
                    };
                }

                // 2. Try UI Automation TextPattern
                if (focused.TryGetCurrentPattern(TextPattern.Pattern, out object patternObj) &&
                    patternObj is TextPattern textPattern)
                {
                    var selection = textPattern.GetSelection();
                    if (selection != null && selection.Length > 0)
                    {
                        var rects = selection[0].GetBoundingRectangles();
                        if (rects != null && rects.Length > 0 && rects[0].Width >= 0)
                        {
                            var r = rects[0];
                            string text = string.Empty;
                            try
                            {
                                // Get surrounding document text before caret (up to ~300 chars)
                                var docRange = textPattern.DocumentRange.Clone();
                                docRange.MoveEndpointByRange(TextPatternRangeEndpoint.End, selection[0], TextPatternRangeEndpoint.Start);
                                string fullBefore = docRange.GetText(-1);
                                text = fullBefore.Length > 300 ? fullBefore.Substring(fullBefore.Length - 300) : fullBefore;
                            }
                            catch
                            {
                                // fallback if document range move fails
                            }

                            return new CaretInfo
                            {
                                IsVisible = true,
                                ScreenPosition = new Point(r.Left, r.Bottom),
                                CaretHeight = Math.Max(16, r.Height),
                                SurroundingText = text,
                                IsPassword = false
                            };
                        }
                    }
                }
            }
        }
        catch
        {
            // UI Automation may throw COMException if app is unresponsive
        }

        // 3. Fallback: Win32 GetGUIThreadInfo
        return GetWin32CaretInfo();
    }

    private static CaretInfo GetWin32CaretInfo()
    {
        IntPtr hwnd = Win32Native.GetForegroundWindow();
        if (hwnd == IntPtr.Zero)
        {
            return new CaretInfo { IsVisible = false, ScreenPosition = new Point(0, 0) };
        }

        uint threadId = Win32Native.GetWindowThreadProcessId(hwnd, out _);
        var gui = new Win32Native.GUITHREADINFO();
        gui.cbSize = Marshal.SizeOf(gui);

        if (Win32Native.GetGUIThreadInfo(threadId, ref gui) && gui.hwndCaret != IntPtr.Zero)
        {
            var pt = new Win32Native.POINT
            {
                X = gui.rcCaret.Left,
                Y = gui.rcCaret.Bottom
            };

            Win32Native.ClientToScreen(gui.hwndCaret, ref pt);
            int height = Math.Max(16, gui.rcCaret.Bottom - gui.rcCaret.Top);

            return new CaretInfo
            {
                IsVisible = true,
                ScreenPosition = new Point(pt.X, pt.Y),
                CaretHeight = height,
                SurroundingText = null, // Will use keyboard buffer fallback
                IsPassword = false
            };
        }

        return new CaretInfo { IsVisible = false, ScreenPosition = new Point(0, 0) };
    }
}
