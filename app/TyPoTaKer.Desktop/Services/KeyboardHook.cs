using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

namespace TyPoTaKer.Desktop.Services;

public sealed class TypingEventArgs : EventArgs
{
    public string BufferText { get; }
    public char? LastChar { get; }

    public TypingEventArgs(string bufferText, char? lastChar)
    {
        BufferText = bufferText;
        LastChar = lastChar;
    }
}

public sealed class KeyboardHook : IDisposable
{
    private readonly Win32Native.LowLevelKeyboardProc _proc;
    private IntPtr _hookId = IntPtr.Zero;
    private readonly StringBuilder _buffer = new();
    private readonly object _lock = new();

    public event EventHandler<TypingEventArgs>? TypingActivity;
    public event EventHandler? AcceptRequested;
    public event EventHandler? DismissRequested;

    public bool IsEnabled { get; set; } = true;

    public KeyboardHook()
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
            _hookId = Win32Native.SetWindowsHookEx(Win32Native.WH_KEYBOARD_LL, _proc, moduleHandle, 0);
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

    public string GetBuffer()
    {
        lock (_lock)
        {
            return _buffer.ToString();
        }
    }

    public void ClearBuffer()
    {
        lock (_lock)
        {
            _buffer.Clear();
        }
    }

    private IntPtr HookCallback(int nCode, IntPtr wParam, IntPtr lParam)
    {
        if (nCode >= 0 && IsEnabled)
        {
            int msg = wParam.ToInt32();
            if (msg == Win32Native.WM_KEYDOWN || msg == Win32Native.WM_SYSKEYDOWN)
            {
                var kbd = Marshal.PtrToStructure<Win32Native.KBDLLHOOKSTRUCT>(lParam);
                uint vk = kbd.vkCode;
                bool isCtrl = (Win32Native.GetAsyncKeyState(Win32Native.VK_CONTROL) & 0x8000) != 0;

                // 1. Accept Hotkey: Ctrl + Space
                if (isCtrl && vk == Win32Native.VK_SPACE)
                {
                    AcceptRequested?.Invoke(this, EventArgs.Empty);
                    return (IntPtr)1; // Consume space key
                }

                // 2. Dismiss Hotkey: Escape
                if (vk == Win32Native.VK_ESCAPE)
                {
                    DismissRequested?.Invoke(this, EventArgs.Empty);
                }
                else if (vk == Win32Native.VK_RETURN)
                {
                    ClearBuffer();
                    DismissRequested?.Invoke(this, EventArgs.Empty);
                }
                else if (vk == Win32Native.VK_BACK)
                {
                    lock (_lock)
                    {
                        if (_buffer.Length > 0)
                            _buffer.Remove(_buffer.Length - 1, 1);
                    }
                    TypingActivity?.Invoke(this, new TypingEventArgs(GetBuffer(), null));
                }
                else if (!isCtrl)
                {
                    char? c = TranslateVkToChar(vk, kbd.scanCode);
                    if (c.HasValue && !char.IsControl(c.Value))
                    {
                        lock (_lock)
                        {
                            _buffer.Append(c.Value);
                            if (_buffer.Length > 250)
                                _buffer.Remove(0, _buffer.Length - 250);
                        }
                        TypingActivity?.Invoke(this, new TypingEventArgs(GetBuffer(), c.Value));
                    }
                }
            }
        }

        return Win32Native.CallNextHookEx(_hookId, nCode, wParam, lParam);
    }

    private static char? TranslateVkToChar(uint vk, uint scanCode)
    {
        byte[] keyState = new byte[256];
        if (!Win32Native.GetKeyboardState(keyState))
            return null;

        IntPtr hwnd = Win32Native.GetForegroundWindow();
        uint threadId = Win32Native.GetWindowThreadProcessId(hwnd, out _);
        IntPtr hkl = Win32Native.GetKeyboardLayout(threadId);

        var sb = new StringBuilder(4);
        int rc = Win32Native.ToUnicodeEx(vk, scanCode, keyState, sb, sb.Capacity, 0, hkl);
        if (rc > 0 && sb.Length > 0)
        {
            return sb[0];
        }
        return null;
    }

    /// <summary>
    /// Simulates keystrokes via Win32 SendInput to replace text directly in the active focused element.
    /// </summary>
    public static void SendText(string text, int backspacesCount = 0)
    {
        var inputs = new System.Collections.Generic.List<Win32Native.INPUT>();

        // 1. Backspaces
        for (int i = 0; i < backspacesCount; i++)
        {
            inputs.Add(new Win32Native.INPUT
            {
                type = Win32Native.INPUT_KEYBOARD,
                ki = new Win32Native.KEYBDINPUT { wVk = Win32Native.VK_BACK }
            });
            inputs.Add(new Win32Native.INPUT
            {
                type = Win32Native.INPUT_KEYBOARD,
                ki = new Win32Native.KEYBDINPUT { wVk = Win32Native.VK_BACK, dwFlags = Win32Native.KEYEVENTF_KEYUP }
            });
        }

        // 2. Unicode characters
        foreach (char c in text)
        {
            inputs.Add(new Win32Native.INPUT
            {
                type = Win32Native.INPUT_KEYBOARD,
                ki = new Win32Native.KEYBDINPUT { wScan = (ushort)c, dwFlags = Win32Native.KEYEVENTF_UNICODE }
            });
            inputs.Add(new Win32Native.INPUT
            {
                type = Win32Native.INPUT_KEYBOARD,
                ki = new Win32Native.KEYBDINPUT { wScan = (ushort)c, dwFlags = Win32Native.KEYEVENTF_UNICODE | Win32Native.KEYEVENTF_KEYUP }
            });
        }

        if (inputs.Count > 0)
        {
            var arr = inputs.ToArray();
            Win32Native.SendInput((uint)arr.Length, arr, Marshal.SizeOf<Win32Native.INPUT>());
        }
    }

    public void Dispose()
    {
        Stop();
    }
}
