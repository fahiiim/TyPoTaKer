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

    public string CurrentInsertion { get; private set; } = string.Empty;
    public int CurrentBackspaces { get; private set; } = 0;
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
        Point caretScreenPos,
        double caretHeight,
        string displayLabel,
        string insertionText,
        bool isCorrection,
        int backspaces = 0)
    {
        Dispatcher.Invoke(() =>
        {
            CurrentInsertion = insertionText;
            CurrentBackspaces = backspaces;
            HasActiveSuggestion = true;

            SuggestionText.Text = displayLabel;

            var bc = new BrushConverter();
            if (isCorrection)
            {
                ModeBadge.Background = (Brush)bc.ConvertFromString("#10B981")!;
                ModeText.Text = "⚡ FIX";
                PillBorder.BorderBrush = (Brush)bc.ConvertFromString("#34D399")!;
            }
            else
            {
                ModeBadge.Background = (Brush)bc.ConvertFromString("#6366F1")!;
                ModeText.Text = "✨ NEXT";
                PillBorder.BorderBrush = (Brush)bc.ConvertFromString("#818CF8")!;
            }

            // Position calculation with offset below caret
            double targetX = caretScreenPos.X;
            double targetY = caretScreenPos.Y + caretHeight + 4;

            // Clamping to screen boundaries
            double screenW = SystemParameters.PrimaryScreenWidth;
            double screenH = SystemParameters.PrimaryScreenHeight;

            if (targetX + 280 > screenW)
                targetX = Math.Max(10, screenW - 300);
            if (targetY + 60 > screenH)
                targetY = Math.Max(10, caretScreenPos.Y - 45); // Flip above caret if near bottom of screen

            Left = Math.Max(10, targetX);
            Top = Math.Max(10, targetY);

            Show();

            // Restart 5-second auto-dismiss timer
            _autoHideTimer.Stop();
            _autoHideTimer.Start();
        });
    }

    public void HideOverlay()
    {
        Dispatcher.Invoke(() =>
        {
            _autoHideTimer.Stop();
            HasActiveSuggestion = false;
            CurrentInsertion = string.Empty;
            CurrentBackspaces = 0;
            Hide();
        });
    }

    private void OnDismissClick(object sender, MouseButtonEventArgs e)
    {
        HideOverlay();
        DismissClicked?.Invoke(this, EventArgs.Empty);
    }
}
