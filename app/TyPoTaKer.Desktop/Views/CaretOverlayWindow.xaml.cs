using System;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Input;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Threading;
using Point = System.Windows.Point;
using Brush = System.Windows.Media.Brush;

namespace TyPoTaKer.Desktop.Views;

public enum SuggestionBadgeType
{
    Fix,
    Next,
    Clean
}

public partial class CaretOverlayWindow : Window
{
    private const int GWL_EXSTYLE = -20;
    private const int WS_EX_NOACTIVATE = 0x08000000;
    private const int WS_EX_TOOLWINDOW = 0x00000080;

    [DllImport("user32.dll", SetLastError = true)]
    private static extern int GetWindowLong(IntPtr hWnd, int nIndex);

    [DllImport("user32.dll", SetLastError = true)]
    private static extern int SetWindowLong(IntPtr hWnd, int nIndex, int dwNewLong);

    private readonly DispatcherTimer _autoHideTimer;

    public event EventHandler? DismissClicked;
    public event EventHandler? SuggestionAccepted;

    public string CurrentInsertion { get; private set; } = string.Empty;
    public int CurrentBackspaces { get; private set; } = 0;
    public bool IsSelectionReplacement { get; private set; } = false;
    public bool HasActiveSuggestion { get; private set; } = false;

    public CaretOverlayWindow()
    {
        InitializeComponent();
        _autoHideTimer = new DispatcherTimer { Interval = TimeSpan.FromSeconds(5) };
        _autoHideTimer.Tick += (s, e) =>
        {
            _autoHideTimer.Stop();
            HideOverlay();
        };

        Loaded += (s, e) => Hide();
    }

    protected override void OnSourceInitialized(EventArgs e)
    {
        base.OnSourceInitialized(e);

        // Ensure this overlay window never steals keyboard focus from active document
        var helper = new WindowInteropHelper(this);
        int exStyle = GetWindowLong(helper.Handle, GWL_EXSTYLE);
        SetWindowLong(helper.Handle, GWL_EXSTYLE, exStyle | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW);
    }

    public void ShowSuggestion(
        Point screenPos,
        double heightOffset,
        string displayLabel,
        string insertionText,
        SuggestionBadgeType badgeType,
        int backspaces = 0,
        bool isSelectionReplacement = false,
        bool isAbovePosition = false)
    {
        Dispatcher.Invoke(() =>
        {
            CurrentInsertion = insertionText;
            CurrentBackspaces = backspaces;
            IsSelectionReplacement = isSelectionReplacement;
            HasActiveSuggestion = badgeType != SuggestionBadgeType.Clean;

            SuggestionText.Text = displayLabel;
            HotkeyBadge.Visibility = Visibility.Visible;

            var bc = new BrushConverter();
            switch (badgeType)
            {
                case SuggestionBadgeType.Fix:
                    ModeBadge.Background = (Brush)bc.ConvertFromString("#10B981")!;
                    ModeText.Text = "⚡ FIX";
                    PillBorder.BorderBrush = (Brush)bc.ConvertFromString("#34D399")!;
                    HotkeyText.Text = "Ctrl + Space";
                    break;
                case SuggestionBadgeType.Next:
                    ModeBadge.Background = (Brush)bc.ConvertFromString("#6366F1")!;
                    ModeText.Text = "✨ NEXT";
                    PillBorder.BorderBrush = (Brush)bc.ConvertFromString("#818CF8")!;
                    HotkeyText.Text = "Ctrl + Space";
                    break;
                case SuggestionBadgeType.Clean:
                    ModeBadge.Background = (Brush)bc.ConvertFromString("#059669")!;
                    ModeText.Text = "✓ CLEAN";
                    PillBorder.BorderBrush = (Brush)bc.ConvertFromString("#10B981")!;
                    HotkeyBadge.Visibility = Visibility.Collapsed;
                    break;
            }

            // Calculate position
            double targetX = screenPos.X;
            double targetY = isAbovePosition
                ? (screenPos.Y - 45) // Float slightly above cursor/menu
                : (screenPos.Y + heightOffset + 4); // Float below caret

            // Clamping to screen boundaries
            double screenW = SystemParameters.PrimaryScreenWidth;
            double screenH = SystemParameters.PrimaryScreenHeight;

            if (targetX + 320 > screenW)
                targetX = Math.Max(10, screenW - 340);
            if (targetY + 60 > screenH)
                targetY = Math.Max(10, screenPos.Y - 45);
            if (targetY < 10)
                targetY = Math.Min(screenH - 60, screenPos.Y + 28);

            Left = Math.Max(10, targetX);
            Top = Math.Max(10, targetY);

            Show();

            // Set auto-hide interval (1.8s for clean status, 5s for suggestions)
            _autoHideTimer.Stop();
            _autoHideTimer.Interval = (badgeType == SuggestionBadgeType.Clean)
                ? TimeSpan.FromSeconds(1.8)
                : TimeSpan.FromSeconds(5.0);
            _autoHideTimer.Start();
        });
    }

    public void ShowCleanStatus(Point screenPos, string message = "0 changes needed")
    {
        ShowSuggestion(
            screenPos: screenPos,
            heightOffset: 0,
            displayLabel: message,
            insertionText: string.Empty,
            badgeType: SuggestionBadgeType.Clean,
            backspaces: 0,
            isSelectionReplacement: false,
            isAbovePosition: true
        );
    }

    public void HideOverlay()
    {
        Dispatcher.Invoke(() =>
        {
            _autoHideTimer.Stop();
            HasActiveSuggestion = false;
            CurrentInsertion = string.Empty;
            CurrentBackspaces = 0;
            IsSelectionReplacement = false;
            Hide();
        });
    }

    private void OnContentClick(object sender, MouseButtonEventArgs e)
    {
        if (HasActiveSuggestion)
        {
            SuggestionAccepted?.Invoke(this, EventArgs.Empty);
        }
    }

    private void OnDismissClick(object sender, MouseButtonEventArgs e)
    {
        HideOverlay();
        DismissClicked?.Invoke(this, EventArgs.Empty);
    }
}
