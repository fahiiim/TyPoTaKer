using System;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Windows;
using Microsoft.Win32;

namespace TyPoTaKer.Desktop.Views;

public partial class DashboardWindow : Window
{
    private const string RUN_REG_KEY = @"Software\Microsoft\Windows\CurrentVersion\Run";
    private const string APP_NAME = "TypoFormer";

    public event EventHandler? ResetBufferRequested;
    public event EventHandler<bool>? TabAcceptToggled;

    public DashboardWindow()
    {
        InitializeComponent();
        CheckStartupState();
    }

    private void CheckStartupState()
    {
        try
        {
            using var key = Registry.CurrentUser.OpenSubKey(RUN_REG_KEY, false);
            var val = key?.GetValue(APP_NAME);
            StartupCheckBox.IsChecked = val != null;
        }
        catch
        {
            StartupCheckBox.IsChecked = false;
        }
    }

    private void OnStartupChecked(object sender, RoutedEventArgs e)
    {
        try
        {
            string exePath = Process.GetCurrentProcess().MainModule?.FileName 
                             ?? Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "TyPoTaKer.Desktop.exe");
            using var key = Registry.CurrentUser.OpenSubKey(RUN_REG_KEY, true);
            key?.SetValue(APP_NAME, $"\"{exePath}\"");
        }
        catch (Exception ex)
        {
            System.Windows.MessageBox.Show($"Failed to set Windows startup: {ex.Message}", "TyPoTaKer", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
    }

    private void OnStartupUnchecked(object sender, RoutedEventArgs e)
    {
        try
        {
            using var key = Registry.CurrentUser.OpenSubKey(RUN_REG_KEY, true);
            key?.DeleteValue(APP_NAME, false);
        }
        catch (Exception ex)
        {
            System.Windows.MessageBox.Show($"Failed to remove Windows startup: {ex.Message}", "TyPoTaKer", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
    }

    private void OnTabAcceptChanged(object sender, RoutedEventArgs e)
    {
        TabAcceptToggled?.Invoke(this, TabAcceptCheckBox.IsChecked == true);
    }

    private void OnResetBufferClick(object sender, RoutedEventArgs e)
    {
        ResetBufferRequested?.Invoke(this, EventArgs.Empty);
        PlaygroundStatusText.Text = "Memory buffer cleared. Ready for fresh typing context.";
    }

    private void OnClearPlaygroundClick(object sender, RoutedEventArgs e)
    {
        PlaygroundTextBox.Text = string.Empty;
        PlaygroundTextBox.Focus();
    }

    private void OnMinimizeClick(object sender, RoutedEventArgs e)
    {
        Hide();
    }

    protected override void OnClosing(CancelEventArgs e)
    {
        // Don't close the application; hide to tray instead
        e.Cancel = true;
        Hide();
    }
}
