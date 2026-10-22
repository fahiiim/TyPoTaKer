using System;
using System.Drawing;
using System.Windows.Forms;

namespace TyPoTaKer.Desktop.Services;

public sealed class TrayManager : IDisposable
{
    private readonly NotifyIcon _notifyIcon;
    private readonly ToolStripMenuItem _enabledItem;

    public event EventHandler<bool>? EnabledToggled;
    public event EventHandler? ResetRequested;
    public event EventHandler? ExitRequested;

    public bool IsEnabled
    {
        get => _enabledItem.Checked;
        set => _enabledItem.Checked = value;
    }

    public TrayManager()
    {
        var contextMenu = new ContextMenuStrip();

        var header = new ToolStripMenuItem("TyPoTaKer Assistant")
        {
            Enabled = false,
            Font = new Font(Control.DefaultFont, FontStyle.Bold)
        };
        contextMenu.Items.Add(header);

        contextMenu.Items.Add(new ToolStripSeparator());

        _enabledItem = new ToolStripMenuItem("Enable Suggestions & Fixes")
        {
            Checked = true
        };
        _enabledItem.Click += (s, e) =>
        {
            _enabledItem.Checked = !_enabledItem.Checked;
            EnabledToggled?.Invoke(this, _enabledItem.Checked);
        };
        contextMenu.Items.Add(_enabledItem);

        var resetItem = new ToolStripMenuItem("Clear Context Buffer", null, (s, e) =>
        {
            ResetRequested?.Invoke(this, EventArgs.Empty);
        });
        contextMenu.Items.Add(resetItem);

        contextMenu.Items.Add(new ToolStripSeparator());

        var infoItem = new ToolStripMenuItem("Models: LM INT8 (43MB) + GEC INT8 (37MB)")
        {
            Enabled = false
        };
        contextMenu.Items.Add(infoItem);

        contextMenu.Items.Add(new ToolStripSeparator());

        var exitItem = new ToolStripMenuItem("Exit TyPoTaKer", null, (s, e) =>
        {
            ExitRequested?.Invoke(this, EventArgs.Empty);
        });
        contextMenu.Items.Add(exitItem);

        _notifyIcon = new NotifyIcon
        {
            Icon = SystemIcons.Information,
            Text = "TyPoTaKer — AI Keystroke Companion",
            ContextMenuStrip = contextMenu,
            Visible = true
        };
    }

    public void ShowNotification(string title, string message)
    {
        _notifyIcon.ShowBalloonTip(2000, title, message, ToolTipIcon.Info);
    }

    public void Dispose()
    {
        _notifyIcon.Visible = false;
        _notifyIcon.Dispose();
    }
}
