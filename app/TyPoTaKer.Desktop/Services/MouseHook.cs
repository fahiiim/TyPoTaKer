using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Point = System.Windows.Point;

namespace TyPoTaKer.Desktop.Services;

public sealed class RightClickSelectionEventArgs : EventArgs
{
    public Point ScreenPosition { get; }
    public bool HasLikelySelection { get; }

    public RightClickSelectionEventArgs(Point screenPosition, bool hasLikelySelection)
    {
        ScreenPosition = screenPosition;
        HasLikelySelection = hasLikelySelection;
    }
}

public sealed class MouseHook : IDisposable
{
    private readonly Win32Native.LowLevelMouseProc _proc;
    private IntPtr _hookId = IntPtr.Zero;

    private Win32Native.POINT _downPt;
    private long _downTime;
    private long _lastClickUpTime;
    private Win32Native.POINT _lastClickUpPt;
    private int _clickCount = 0;

    private bool _hasSelection = false;
    private long _lastSelectionTime = 0;
    private Point _lastSelectionScreenPt = new(0, 0);

    public event EventHandler<RightClickSelectionEventArgs>? RightClickSelectionRequested;

    public bool IsEnabled { get; set; } = true;

    public bool HasLikelySelection
    {
        get
        {
            if (!_hasSelection) return false;
            // Valid within 15 seconds of mouse drag or double/triple click
            return (Environment.TickCount64 - _lastSelectionTime) <= 15000;
        }
        set
        {
            _hasSelection = value;
            if (value)
            {
                _lastSelectionTime = Environment.TickCount64;
            }
        }
    }

    public MouseHook()
    {
        _proc = HookCallback;
    }

    public void Start()
    {
        if (_hookId == IntPtr.Zero)
        {
            using var curProcess = Process.GetCurrentProcess();
            using var curModule = curProcess.MainModule;
            IntPtr moduleHandle = curModule?.ModuleName != null ? Win32Native.GetModuleHandle(curModule.ModuleName) : IntPtr.Zero;
            _hookId = Win32Native.SetWindowsHookEx(Win32Native.WH_MOUSE_LL, _proc, moduleHandle, 0);
        }
    }

    public void Stop()
    {
        if (_hookId != IntPtr.Zero)
        {
            Win32Native.UnhookWindowsHookEx(_hookId);
            _hookId = IntPtr.Zero;
        }
    }

    public void NotifyExternalSelection()
    {
        _hasSelection = true;
        _lastSelectionTime = Environment.TickCount64;
    }

    public void ClearSelection()
    {
        _hasSelection = false;
    }

    private IntPtr HookCallback(int nCode, IntPtr wParam, IntPtr lParam)
    {
        if (nCode >= 0 && IsEnabled)
        {
            int msg = wParam.ToInt32();
            var mouse = Marshal.PtrToStructure<Win32Native.MSLLHOOKSTRUCT>(lParam);

            if (msg == Win32Native.WM_LBUTTONDOWN)
            {
                _downPt = mouse.pt;
                _downTime = Environment.TickCount64;

                uint dblTime = Win32Native.GetDoubleClickTime();
                long elapsedSinceLastUp = _downTime - _lastClickUpTime;

                int cdx = mouse.pt.X - _lastClickUpPt.X;
                int cdy = mouse.pt.Y - _lastClickUpPt.Y;
                bool isClose = (cdx * cdx + cdy * cdy) <= 25;

                if (elapsedSinceLastUp <= dblTime && isClose)
                {
                    _clickCount++;
                    // Double or triple click selects word or line
                    if (_clickCount >= 2)
                    {
                        _hasSelection = true;
                        _lastSelectionTime = _downTime;
                        _lastSelectionScreenPt = new Point(mouse.pt.X, mouse.pt.Y);
                    }
                }
                else
                {
                    _clickCount = 1;
                }
            }
            else if (msg == Win32Native.WM_LBUTTONUP)
            {
                long now = Environment.TickCount64;
                _lastClickUpTime = now;
                _lastClickUpPt = mouse.pt;

                int dx = mouse.pt.X - _downPt.X;
                int dy = mouse.pt.Y - _downPt.Y;

                // If user dragged mouse by > 5 pixels, text was highlighted/selected
                if ((dx * dx + dy * dy) >= 25)
                {
                    _hasSelection = true;
                    _lastSelectionTime = now;
                    _lastSelectionScreenPt = new Point(mouse.pt.X, mouse.pt.Y);
                }
                else if (_clickCount < 2)
                {
                    // Single simple click without dragging clears selection state
                    _hasSelection = false;
                }
            }
            else if (msg == Win32Native.WM_RBUTTONUP)
            {
                var pt = new Point(mouse.pt.X, mouse.pt.Y);
                bool hasSel = HasLikelySelection;

                RightClickSelectionRequested?.Invoke(
                    this,
                    new RightClickSelectionEventArgs(pt, hasSel)
                );
            }
        }

        return Win32Native.CallNextHookEx(_hookId, nCode, wParam, lParam);
    }

    public void Dispose()
    {
        Stop();
    }
}
