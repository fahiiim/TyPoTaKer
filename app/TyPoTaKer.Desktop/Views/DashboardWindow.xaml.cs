using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Threading;
using Microsoft.Win32;
using TyPoTaKer.Desktop.Engine;
using Point = System.Windows.Point;
using Color = System.Windows.Media.Color;
using ColorConverter = System.Windows.Media.ColorConverter;
using Brushes = System.Windows.Media.Brushes;
using Brush = System.Windows.Media.Brush;
using Pen = System.Windows.Media.Pen;
using MouseEventArgs = System.Windows.Input.MouseEventArgs;
using KeyEventArgs = System.Windows.Input.KeyEventArgs;

namespace TyPoTaKer.Desktop.Views;

public sealed class GrammarlyErrorItem
{
    public string Original { get; set; } = string.Empty;
    public string Replacement { get; set; } = string.Empty;
    public string Category { get; set; } = "Spelling";
    public TextRange? Range { get; set; }
}

public partial class DashboardWindow : Window
{
    private const string RUN_REG_KEY = @"Software\Microsoft\Windows\CurrentVersion\Run";
    private const string APP_NAME = "TypoFormer";

    private TyPoInferenceEngine? _engine;
    private readonly DispatcherTimer _debounceTimer;
    private readonly List<GrammarlyErrorItem> _detectedErrors = new();
    private GrammarlyErrorItem? _currentActiveError;
    private bool _isApplyingFix = false;
    private CancellationTokenSource? _analysisCts;

    public event EventHandler? ResetBufferRequested;
    public event EventHandler<bool>? TabAcceptToggled;

    public DashboardWindow()
    {
        InitializeComponent();
        CheckStartupState();

        _debounceTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(250) };
        _debounceTimer.Tick += OnDebounceElapsed;

        Loaded += (s, e) =>
        {
            // Initial analysis on default text
            TriggerAnalysis();
        };
    }

    public void SetEngine(TyPoInferenceEngine engine)
    {
        _engine = engine;
        TriggerAnalysis();
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
            System.Windows.MessageBox.Show($"Failed to set Windows startup: {ex.Message}", "TypoFormer", MessageBoxButton.OK, MessageBoxImage.Warning);
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
            System.Windows.MessageBox.Show($"Failed to remove Windows startup: {ex.Message}", "TypoFormer", MessageBoxButton.OK, MessageBoxImage.Warning);
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
        _isApplyingFix = true;
        try
        {
            PlaygroundRichTextBox.Document.Blocks.Clear();
            PlaygroundRichTextBox.Document.Blocks.Add(new Paragraph());
            _detectedErrors.Clear();
            GrammarlyPopup.IsOpen = false;
            UpdateErrorBadge();
            PlaygroundRichTextBox.Focus();
        }
        finally
        {
            _isApplyingFix = false;
        }
    }

    private void OnMinimizeClick(object sender, RoutedEventArgs e)
    {
        Hide();
    }

    protected override void OnClosing(CancelEventArgs e)
    {
        e.Cancel = true;
        Hide();
    }

    private void OnPlaygroundTextChanged(object sender, TextChangedEventArgs e)
    {
        if (_isApplyingFix) return;
        _debounceTimer.Stop();
        _debounceTimer.Start();
    }

    private void OnDebounceElapsed(object? sender, EventArgs e)
    {
        _debounceTimer.Stop();
        TriggerAnalysis();
    }

    private void TriggerAnalysis()
    {
        if (_engine == null) return;

        _analysisCts?.Cancel();
        var cts = new CancellationTokenSource();
        _analysisCts = cts;
        var ct = cts.Token;

        string plainText = GetPlainText();
        if (string.IsNullOrWhiteSpace(plainText) || plainText.Length < 3)
        {
            ClearAllHighlights();
            return;
        }

        Task.Run(() =>
        {
            if (ct.IsCancellationRequested) return;

            // Split into sentences and analyze each
            var sentences = plainText.Split(new[] { '.', '!', '?', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
            var allEdits = new List<(string Original, string Replacement, float Confidence)>();

            foreach (var s in sentences)
            {
                if (ct.IsCancellationRequested) return;
                string trimmed = s.Trim();
                if (trimmed.Length < 3) continue;

                var res = _engine.CorrectSentence(trimmed, minEditProb: 0.45f, editMargin: 0.05f);
                if (res.HasCorrection && res.Edits.Count > 0)
                {
                    allEdits.AddRange(res.Edits);
                }
            }

            if (ct.IsCancellationRequested) return;

            Dispatcher.Invoke(() =>
            {
                if (ct.IsCancellationRequested) return;
                HighlightDetectedErrors(allEdits);
            });
        }, ct);
    }

    private string GetPlainText()
    {
        var doc = PlaygroundRichTextBox.Document;
        return new TextRange(doc.ContentStart, doc.ContentEnd).Text.Trim();
    }

    private void ClearAllHighlights()
    {
        _detectedErrors.Clear();
        GrammarlyPopup.IsOpen = false;
        UpdateErrorBadge();

        foreach (var block in PlaygroundRichTextBox.Document.Blocks)
        {
            if (block is Paragraph p)
            {
                foreach (var inline in p.Inlines)
                {
                    inline.TextDecorations = null;
                    inline.Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#F8FAFC"));
                    inline.Background = Brushes.Transparent;
                }
            }
        }
    }

    private void HighlightDetectedErrors(List<(string Original, string Replacement, float Confidence)> edits)
    {
        if (_isApplyingFix) return;

        ClearAllHighlights();
        if (edits == null || edits.Count == 0) return;

        var redBrush = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#EF4444"));
        var softRedBg = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#2E1010"));
        var softRedFg = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#FCA5A5"));

        var wavyUnderline = new TextDecoration
        {
            Location = TextDecorationLocation.Underline,
            Pen = new Pen(redBrush, 1.8) { DashStyle = DashStyles.Dot }
        };
        var decs = new TextDecorationCollection { wavyUnderline };

        foreach (var edit in edits)
        {
            string orig = edit.Original.Trim();
            string repl = edit.Replacement.Trim();
            if (string.IsNullOrEmpty(orig) || string.IsNullOrEmpty(repl) || orig.Equals(repl, StringComparison.OrdinalIgnoreCase))
                continue;

            string cat = GuessCategory(orig, repl);

            // Strip trailing/leading punctuation to isolate the exact word
            string cleanOrig = orig.Trim(".,!?:;\"'()[]{}<>-/\\".ToCharArray());
            string cleanRepl = repl.Trim(".,!?:;\"'()[]{}<>-/\\".ToCharArray());
            if (string.IsNullOrEmpty(cleanOrig) || string.IsNullOrEmpty(cleanRepl) || cleanOrig.Equals(cleanRepl, StringComparison.OrdinalIgnoreCase))
                continue;

            // Find matching word in FlowDocument with whole-word boundary
            TextRange? range = FindWordRangeInDocument(PlaygroundRichTextBox.Document, cleanOrig, _detectedErrors);
            if (range != null)
            {
                range.ApplyPropertyValue(Inline.TextDecorationsProperty, decs);
                range.ApplyPropertyValue(TextElement.ForegroundProperty, softRedFg);
                range.ApplyPropertyValue(TextElement.BackgroundProperty, softRedBg);

                _detectedErrors.Add(new GrammarlyErrorItem
                {
                    Original = cleanOrig,
                    Replacement = cleanRepl,
                    Category = cat,
                    Range = range
                });
            }
        }

        UpdateErrorBadge();
        CheckCursorPositionForErrors();
    }

    private void UpdateErrorBadge()
    {
        if (_detectedErrors.Count > 0)
        {
            ErrorCounterBadge.Visibility = Visibility.Visible;
            ErrorCounterText.Text = $"{_detectedErrors.Count} suggestion{(_detectedErrors.Count == 1 ? "" : "s")}";
        }
        else
        {
            ErrorCounterBadge.Visibility = Visibility.Collapsed;
        }
    }

    private static TextRange? FindWordRangeInDocument(FlowDocument doc, string searchText, List<GrammarlyErrorItem> existingErrors)
    {
        for (TextPointer start = doc.ContentStart; start != null && start.CompareTo(doc.ContentEnd) < 0; start = start.GetNextContextPosition(LogicalDirection.Forward))
        {
            if (start.GetPointerContext(LogicalDirection.Forward) == TextPointerContext.Text)
            {
                string text = start.GetTextInRun(LogicalDirection.Forward);
                int searchIndex = 0;
                while (searchIndex < text.Length)
                {
                    int index = text.IndexOf(searchText, searchIndex, StringComparison.OrdinalIgnoreCase);
                    if (index == -1) break;

                    // Ensure whole-word boundary before and after
                    bool boundaryBefore = (index == 0) || !char.IsLetterOrDigit(text[index - 1]);
                    bool boundaryAfter = (index + searchText.Length >= text.Length) || !char.IsLetterOrDigit(text[index + searchText.Length]);

                    if (boundaryBefore && boundaryAfter)
                    {
                        TextPointer p1 = start.GetPositionAtOffset(index);
                        TextPointer p2 = p1.GetPositionAtOffset(searchText.Length);

                        // Overlap check with existing errors
                        bool overlaps = existingErrors.Any(e => e.Range != null &&
                            ((e.Range.Start.CompareTo(p1) <= 0 && e.Range.End.CompareTo(p1) > 0) ||
                             (e.Range.Start.CompareTo(p2) < 0 && e.Range.End.CompareTo(p2) >= 0) ||
                             (p1.CompareTo(e.Range.Start) <= 0 && p2.CompareTo(e.Range.End) >= 0)));

                        if (!overlaps)
                        {
                            return new TextRange(p1, p2);
                        }
                    }

                    searchIndex = index + 1;
                }
            }
        }
        return null;
    }

    private static string GuessCategory(string orig, string repl)
    {
        string o = orig.ToLowerInvariant();
        string r = repl.ToLowerInvariant();
        if ((o == "were" && r == "was") || (o == "was" && r == "were") || (o == "have" && r == "has") || (o == "has" && r == "have") || (o == "is" && r == "are") || (o == "are" && r == "is"))
            return "Agreement";
        if ((o == "writes" && r == "wrote") || (o == "wrote" && r == "writes") || (o == "run" && r == "ran") || (o == "see" && r == "saw"))
            return "Verb Tense";
        if (o == "on" || o == "in" || o == "at" || o == "to" || o == "for" || o == "with" || o == "by" || o == "of")
            return "Preposition";
        if ((o == "an" && r == "a") || (o == "a" && r == "an") || (o == "the" && r == "a"))
            return "Article";
        if (o == "their" && r.Contains("they") || (o == "loose" && r == "lose") || (o == "its" && r.Contains("it")) || (o == "effect" && r == "affect"))
            return "Confusable";
        return "Spelling";
    }

    private void OnPlaygroundSelectionChanged(object sender, RoutedEventArgs e)
    {
        CheckCursorPositionForErrors();
    }

    private void OnPlaygroundMouseMove(object sender, System.Windows.Input.MouseEventArgs e)
    {
        if (GrammarlyPopup.IsOpen && GrammarlyPopup.IsMouseOver) return;

        Point mousePos = e.GetPosition(PlaygroundRichTextBox);
        TextPointer? pointer = PlaygroundRichTextBox.GetPositionFromPoint(mousePos, snapToText: false);
        if (pointer != null)
        {
            GrammarlyErrorItem? hovered = _detectedErrors.FirstOrDefault(err => err.Range != null && err.Range.Contains(pointer));
            if (hovered != null)
            {
                ShowGrammarlyCard(hovered);
                return;
            }
        }

        // If not hovering over an error word or card, close card unless caret is on it
        TextPointer caret = PlaygroundRichTextBox.CaretPosition;
        if (caret == null || !_detectedErrors.Any(err => err.Range != null && err.Range.Contains(caret)))
        {
            if (!GrammarlyPopup.IsMouseOver)
            {
                GrammarlyPopup.IsOpen = false;
            }
        }
    }

    private void CheckCursorPositionForErrors()
    {
        if (_isApplyingFix || _detectedErrors.Count == 0)
        {
            GrammarlyPopup.IsOpen = false;
            return;
        }

        TextPointer caret = PlaygroundRichTextBox.CaretPosition;
        if (caret == null) return;

        GrammarlyErrorItem? matched = _detectedErrors.FirstOrDefault(err => err.Range != null && err.Range.Contains(caret));
        if (matched != null)
        {
            ShowGrammarlyCard(matched);
        }
        else if (!GrammarlyPopup.IsMouseOver)
        {
            GrammarlyPopup.IsOpen = false;
        }
    }

    private void ShowGrammarlyCard(GrammarlyErrorItem error)
    {
        _currentActiveError = error;

        CardOriginalText.Text = error.Original;
        CardReplacementText.Text = error.Replacement;
        CardCategoryText.Text = error.Category.ToUpperInvariant();

        var bc = new BrushConverter();
        if (error.Category == "Spelling")
        {
            CardCategoryBadge.Background = (Brush)bc.ConvertFromString("#7F1D1D")!;
            CardCategoryText.Foreground = (Brush)bc.ConvertFromString("#FCA5A5")!;
        }
        else if (error.Category == "Agreement" || error.Category == "Verb Tense")
        {
            CardCategoryBadge.Background = (Brush)bc.ConvertFromString("#78350F")!;
            CardCategoryText.Foreground = (Brush)bc.ConvertFromString("#FDE68A")!;
        }
        else
        {
            CardCategoryBadge.Background = (Brush)bc.ConvertFromString("#1E1B4B")!;
            CardCategoryText.Foreground = (Brush)bc.ConvertFromString("#C7D2FE")!;
        }

        if (error.Range?.Start != null)
        {
            Rect charRect = error.Range.Start.GetCharacterRect(LogicalDirection.Forward);
            double hOffset = Math.Max(10, charRect.Left - 10);
            if (PlaygroundRichTextBox.ActualWidth > 320)
            {
                hOffset = Math.Min(hOffset, PlaygroundRichTextBox.ActualWidth - 320);
            }
            GrammarlyPopup.HorizontalOffset = hOffset;

            // Position card smartly above or below the target word
            if (charRect.Bottom + 130 > PlaygroundRichTextBox.ActualHeight && charRect.Top > 120)
            {
                GrammarlyPopup.VerticalOffset = Math.Max(10, charRect.Top - 118);
            }
            else
            {
                GrammarlyPopup.VerticalOffset = charRect.Bottom + 6;
            }

            GrammarlyPopup.IsOpen = true;
        }
    }

    private void OnCardApplyClick(object sender, RoutedEventArgs e)
    {
        if (_currentActiveError?.Range != null)
        {
            _isApplyingFix = true;
            try
            {
                // Replace ONLY that specific word at that exact position!
                _currentActiveError.Range.Text = _currentActiveError.Replacement;
                _currentActiveError.Range.ApplyPropertyValue(Inline.TextDecorationsProperty, null);
                _currentActiveError.Range.ApplyPropertyValue(TextElement.BackgroundProperty, Brushes.Transparent);
                _currentActiveError.Range.ApplyPropertyValue(TextElement.ForegroundProperty, new SolidColorBrush((Color)ColorConverter.ConvertFromString("#F8FAFC")));

                _detectedErrors.Remove(_currentActiveError);
                GrammarlyPopup.IsOpen = false;
                _currentActiveError = null;

                UpdateErrorBadge();
                PlaygroundStatusText.Text = "Correction applied cleanly at target position.";
            }
            finally
            {
                _isApplyingFix = false;
                // Seamlessly trigger re-analysis so any remaining errors are re-aligned
                _debounceTimer.Stop();
                _debounceTimer.Start();
            }
        }
    }

    private void OnDismissCardClick(object sender, RoutedEventArgs e)
    {
        if (_currentActiveError?.Range != null)
        {
            _currentActiveError.Range.ApplyPropertyValue(Inline.TextDecorationsProperty, null);
            _currentActiveError.Range.ApplyPropertyValue(TextElement.BackgroundProperty, Brushes.Transparent);
            _currentActiveError.Range.ApplyPropertyValue(TextElement.ForegroundProperty, new SolidColorBrush((Color)ColorConverter.ConvertFromString("#F8FAFC")));
            _detectedErrors.Remove(_currentActiveError);
            _currentActiveError = null;
            UpdateErrorBadge();
        }
        GrammarlyPopup.IsOpen = false;
    }

    private void OnPlaygroundPreviewKeyDown(object sender, System.Windows.Input.KeyEventArgs e)
    {
        if (GrammarlyPopup.IsOpen && (e.Key == Key.Enter || e.Key == Key.Tab))
        {
            OnCardApplyClick(sender, e);
            e.Handled = true;
        }
    }
}
