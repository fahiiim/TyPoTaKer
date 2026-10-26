using System;
using System.Runtime.InteropServices;
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
        // 1. Ultra-fast path (<0.05 ms): Win32 GetGUIThreadInfo
        var win32Info = GetWin32CaretInfo();
        if (win32Info.IsVisible)
        {
            return win32Info;
        }

        // 2. Fallback path for modern apps (WPF, Chromium, Word, Notepad): UI Automation bounds & text
        try
        {
            var focused = AutomationElement.FocusedElement;
            if (focused != null)
            {
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
                            string? extractedText = null;
                            try
                            {
                                var range = selection[0].Clone();
                                range.ExpandToEnclosingUnit(TextUnit.Line);
                                string lineText = range.GetText(250);
                                if (!string.IsNullOrWhiteSpace(lineText))
                                {
                                    extractedText = lineText;
                                }
                            }
                            catch { }

                            return new CaretInfo
                            {
                                IsVisible = true,
                                ScreenPosition = new Point(r.Left, r.Bottom),
                                CaretHeight = Math.Max(16, r.Height),
                                SurroundingText = extractedText,
                                IsPassword = false
                            };
                        }
                    }
                }
                // 3. Fallback: Focused element bounding rectangle
                var elemRect = focused.Current.BoundingRectangle;
                if (!elemRect.IsEmpty && elemRect.Width > 20 && elemRect.Height > 10)
                {
                    return new CaretInfo
                    {
                        IsVisible = true,
                        ScreenPosition = new Point(elemRect.Left + 24, elemRect.Top + 36),
                        CaretHeight = 20,
                        SurroundingText = null,
                        IsPassword = false
                    };
                }
            }
        }
        catch
        {
            // COM exceptions when switching windows
        }

        // 4. Fallback: Foreground window bounding rectangle or cursor position
        try
        {
            IntPtr fgHwnd = Win32Native.GetForegroundWindow();
            if (fgHwnd != IntPtr.Zero && Win32Native.GetWindowRect(fgHwnd, out var wRect))
            {
                if (wRect.Right - wRect.Left > 60 && wRect.Bottom - wRect.Top > 60)
                {
                    if (Win32Native.GetCursorPos(out var mousePt) &&
                        mousePt.X >= wRect.Left && mousePt.X <= wRect.Right &&
                        mousePt.Y >= wRect.Top && mousePt.Y <= wRect.Bottom)
                    {
                        return new CaretInfo
                        {
                            IsVisible = true,
                            ScreenPosition = new Point(mousePt.X, mousePt.Y + 16),
                            CaretHeight = 20,
                            SurroundingText = null,
                            IsPassword = false
                        };
                    }

                    return new CaretInfo
                    {
                        IsVisible = true,
                        ScreenPosition = new Point(wRect.Left + 50, wRect.Top + 80),
                        CaretHeight = 20,
                        SurroundingText = null,
                        IsPassword = false
                    };
                }
            }
        }
        catch { }

        return win32Info;
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
                SurroundingText = null,
                IsPassword = false
            };
        }

        return new CaretInfo { IsVisible = false, ScreenPosition = new Point(0, 0) };
    }
}
