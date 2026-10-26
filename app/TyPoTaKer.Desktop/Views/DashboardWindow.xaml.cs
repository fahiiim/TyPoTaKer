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
using System.Windows.Media.Effects;
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
using Button = System.Windows.Controls.Button;
using Orientation = System.Windows.Controls.Orientation;
using Cursors = System.Windows.Input.Cursors;
using HorizontalAlignment = System.Windows.HorizontalAlignment;

namespace TyPoTaKer.Desktop.Views;

public sealed class QuillBotErrorItem
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public string Original { get; set; } = string.Empty;
    public string Replacement { get; set; } = string.Empty;
    public string ActionDescription { get; set; } = "Correct the spelling";
    public string Category { get; set; } = "Grammar";
    public TextRange? Range { get; set; }
    public bool IsExpanded { get; set; } = false;
}

public partial class DashboardWindow : Window
{
    private TyPoInferenceEngine? _engine;
    private readonly DispatcherTimer _debounceTimer;
    private readonly List<QuillBotErrorItem> _detectedErrors = new();
    private bool _isApplyingFix = false;
    private CancellationTokenSource? _analysisCts;

    public event EventHandler? ResetBufferRequested;
    public event EventHandler<bool>? TabAcceptToggled;

    public DashboardWindow()
    {
        _debounceTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(200) };
        _debounceTimer.Tick += OnDebounceElapsed;

        InitializeComponent();

        Loaded += (s, e) =>
        {
            Topmost = false;
            Activate();
            Focus();
            TriggerAnalysis();
        };
    }

    public void SetEngine(TyPoInferenceEngine engine)
    {
        _engine = engine;
        TriggerAnalysis();
    }

    private void OnTabAcceptChanged(object sender, RoutedEventArgs e)
    {
        TabAcceptToggled?.Invoke(this, TabAcceptCheckBox.IsChecked == true);
    }

    private void OnClearPlaygroundClick(object sender, RoutedEventArgs e)
    {
        _isApplyingFix = true;
        try
        {
            PlaygroundRichTextBox.Document.Blocks.Clear();
            PlaygroundRichTextBox.Document.Blocks.Add(new Paragraph());
            _detectedErrors.Clear();
            RefreshErrorDisplay();
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

    private void OnCloseSidebarClick(object sender, RoutedEventArgs e)
    {
        SidebarBorder.Visibility = SidebarBorder.Visibility == Visibility.Visible ? Visibility.Collapsed : Visibility.Visible;
    }

    private void OnFloatingFixPillClick(object sender, MouseButtonEventArgs e)
    {
        if (SidebarBorder.Visibility != Visibility.Visible)
        {
            SidebarBorder.Visibility = Visibility.Visible;
        }
        AcceptActiveError();
    }

    protected override void OnClosing(CancelEventArgs e)
    {
        e.Cancel = true;
        Hide();
    }

    private void OnPlaygroundTextChanged(object sender, TextChangedEventArgs e)
    {
        if (_isApplyingFix || _debounceTimer == null) return;
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
        _analysisCts?.Cancel();
        var cts = new CancellationTokenSource();
        _analysisCts = cts;
        var ct = cts.Token;

        string plainText = GetPlainText();
        if (string.IsNullOrWhiteSpace(plainText) || plainText.Length < 2)
        {
            ClearAllHighlights();
            return;
        }

        Task.Run(() =>
        {
            try
            {
                if (ct.IsCancellationRequested) return;

                var candidates = new List<(string Original, string Replacement, string Action, string Category)>();

                // 1. Rule Engine: Capitalization, Mid-Sentence Caps ("Never" -> "never"), Slang/Informal ("Gonna" -> "going to"), Typos
                var ruleEdits = GrammarRuleEngine.FindRuleEdits(plainText);
                foreach (var re in ruleEdits)
                {
                    if (!candidates.Any(c => c.Original.Equals(re.Original, StringComparison.OrdinalIgnoreCase)))
                    {
                        candidates.Add((re.Original, re.Replacement, re.ActionDescription, re.Category));
                    }
                }

                // 2. Sentence Capitalization Analysis (Sentence Boundaries)
                AnalyzeCapitalization(plainText, candidates);

                // 3. GEC & Typo Analysis with TyPoInferenceEngine
                if (_engine != null)
                {
                    var sentences = plainText.Split(new[] { '.', '!', '?', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
                    foreach (var s in sentences)
                    {
                        if (ct.IsCancellationRequested) return;
                        string trimmed = s.Trim();
                        if (trimmed.Length < 2) continue;

                        var res = _engine.CorrectSentence(trimmed, minEditProb: 0.35f, editMargin: 0.02f);
                        if (res.HasCorrection && res.Edits.Count > 0)
                        {
                            foreach (var edit in res.Edits)
                            {
                                string o = edit.Original.Trim(".,!?:;\"'()[]{}<>-/\\".ToCharArray());
                                string r = edit.Replacement.Trim(".,!?:;\"'()[]{}<>-/\\".ToCharArray());

                                if (string.IsNullOrEmpty(o) || string.IsNullOrEmpty(r) || o.Equals(r, StringComparison.OrdinalIgnoreCase))
                                    continue;

                                // Avoid duplicate if already caught by rules
                                if (candidates.Any(c => c.Original.Equals(o, StringComparison.OrdinalIgnoreCase)))
                                    continue;

                                string action = GuessQuillBotAction(o, r);
                                candidates.Add((o, r, action, "Grammar"));
                            }
                        }
                    }
                }

                if (ct.IsCancellationRequested) return;

                Dispatcher.Invoke(() =>
                {
                    if (ct.IsCancellationRequested) return;
                    ApplyErrorCandidates(candidates);
                });
            }
            catch
            {
                // Background cancel or safe non-fatal catch
            }
        }, ct);
    }

    private static void AnalyzeCapitalization(string text, List<(string Original, string Replacement, string Action, string Category)> list)
    {
        // Check beginning of text
        string trimmedStart = text.TrimStart();
        if (trimmedStart.Length > 0 && char.IsLower(trimmedStart[0]))
        {
            string firstWord = GetFirstWord(trimmedStart);
            if (!string.IsNullOrEmpty(firstWord) && char.IsLower(firstWord[0]))
            {
                string cap = char.ToUpperInvariant(firstWord[0]) + (firstWord.Length > 1 ? firstWord.Substring(1) : "");
                list.Add((firstWord, cap, "Correct the capitalization", "Grammar"));
            }
        }

        // Check after sentence boundaries (. ! ?)
        for (int i = 0; i < text.Length - 1; i++)
        {
            char c = text[i];
            if (c == '.' || c == '!' || c == '?')
            {
                int j = i + 1;
                while (j < text.Length && char.IsWhiteSpace(text[j])) j++;
                if (j < text.Length && char.IsLower(text[j]))
                {
                    string word = GetFirstWord(text.Substring(j));
                    if (!string.IsNullOrEmpty(word) && char.IsLower(word[0]))
                    {
                        string cap = char.ToUpperInvariant(word[0]) + (word.Length > 1 ? word.Substring(1) : "");
                        if (!list.Any(x => x.Original.Equals(word, StringComparison.OrdinalIgnoreCase)))
                        {
                            list.Add((word, cap, "Correct the capitalization", "Grammar"));
                        }
                    }
                }
            }
        }
    }

    private static string GetFirstWord(string text)
    {
        int end = 0;
        while (end < text.Length && char.IsLetterOrDigit(text[end])) end++;
        return end > 0 ? text.Substring(0, end) : "";
    }

    private static string GuessQuillBotAction(string orig, string repl)
    {
        string o = orig.ToLowerInvariant();
        string r = repl.ToLowerInvariant();

        if (o == r && char.IsLower(orig[0]) && char.IsUpper(repl[0]))
            return "Correct the capitalization";

        if ((o == "were" && r == "was") || (o == "was" && r == "were") || (o == "si" && r == "is") || (o == "writes" && r == "wrote") || (o == "wrote" && r == "writes") || (o == "is" && r == "was") || (o == "are" && r == "were"))
            return "Change the verb tense";

        if ((o == "have" && r == "has") || (o == "has" && r == "have") || (o == "is" && r == "are") || (o == "are" && r == "is"))
            return "Correct the agreement";

        if (o == "on" || o == "in" || o == "at" || o == "to" || o == "for" || o == "with" || o == "by" || o == "of")
            return "Correct the preposition";

        if ((o == "an" && r == "a") || (o == "a" && r == "an") || (o == "the" && r == "a"))
            return "Correct the article";

        return "Correct the spelling";
    }

    private string GetPlainText()
    {
        var doc = PlaygroundRichTextBox.Document;
        return new TextRange(doc.ContentStart, doc.ContentEnd).Text.Trim();
    }

    private void ClearAllHighlights()
    {
        _detectedErrors.Clear();
        RefreshErrorDisplay();

        _isApplyingFix = true;
        PlaygroundRichTextBox.TextChanged -= OnPlaygroundTextChanged;
        try
        {
            foreach (var block in PlaygroundRichTextBox.Document.Blocks)
            {
                if (block is Paragraph p)
                {
                    foreach (var inline in p.Inlines)
                    {
                        inline.TextDecorations = null;
                        inline.Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#0F172A"));
                        inline.Background = Brushes.Transparent;
                    }
                }
            }
        }
        finally
        {
            PlaygroundRichTextBox.TextChanged += OnPlaygroundTextChanged;
            _isApplyingFix = false;
        }
    }

    private void ApplyErrorCandidates(List<(string Original, string Replacement, string Action, string Category)> candidates)
    {
        if (_isApplyingFix) return;

        _isApplyingFix = true;
        PlaygroundRichTextBox.TextChanged -= OnPlaygroundTextChanged;
        try
        {
            ClearAllHighlights();
            if (candidates == null || candidates.Count == 0) return;

            // QuillBot Red Underline + Soft Pink/Red Highlight
            var redUnderlineBrush = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#DC2626"));
            var softRedBg = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#FEE2E2"));
            var normalTextFg = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#0F172A"));

            var solidUnderline = new TextDecoration
            {
                Location = TextDecorationLocation.Underline,
                Pen = new Pen(redUnderlineBrush, 1.8)
            };
            var decs = new TextDecorationCollection { solidUnderline };

            foreach (var c in candidates)
            {
                TextRange? range = FindWordRangeInDocument(PlaygroundRichTextBox.Document, c.Original, _detectedErrors);
                if (range != null)
                {
                    range.ApplyPropertyValue(Inline.TextDecorationsProperty, decs);
                    range.ApplyPropertyValue(TextElement.BackgroundProperty, softRedBg);
                    range.ApplyPropertyValue(TextElement.ForegroundProperty, normalTextFg);

                    _detectedErrors.Add(new QuillBotErrorItem
                    {
                        Original = c.Original,
                        Replacement = c.Replacement,
                        ActionDescription = c.Action,
                        Category = c.Category,
                        Range = range,
                        IsExpanded = _detectedErrors.Count == 0 // Expand first error card by default!
                    });
                }
            }

            RefreshErrorDisplay();
        }
        finally
        {
            PlaygroundRichTextBox.TextChanged += OnPlaygroundTextChanged;
            _isApplyingFix = false;
        }
    }

    private void RefreshErrorDisplay()
    {
        int count = _detectedErrors.Count;

        // 1. Update Left Gutter Badge (Screenshot 2)
        if (count > 0)
        {
            GutterBadge.Visibility = Visibility.Visible;
            GutterCountText.Text = count.ToString();
        }
        else
        {
            GutterBadge.Visibility = Visibility.Collapsed;
        }

        // 2. Update Floating Pill Widget: "Fix N errors [tab] ▾" (Screenshot 2)
        if (count > 0)
        {
            FloatingFixPill.Visibility = Visibility.Visible;
            FloatingFixText.Text = $"Fix {count} error{(count == 1 ? "" : "s")}";
        }
        else
        {
            FloatingFixPill.Visibility = Visibility.Collapsed;
        }

        // 3. Update Sidebar Header Tab Counters (Screenshot 1)
        AllCountText.Text = count.ToString();
        GrammarCountText.Text = count.ToString();

        // 4. Render QuillBot Cards in Sidebar (Screenshot 1)
        ErrorCardsContainer.Children.Clear();

        if (count == 0)
        {
            EmptyStatePanel.Visibility = Visibility.Visible;
            StatusStripText.Text = "All clear: No grammar or spelling errors detected.";
            return;
        }

        EmptyStatePanel.Visibility = Visibility.Collapsed;
        StatusStripText.Text = $"QuillBot Assistant: {count} suggestion{(count == 1 ? "" : "s")} found.";

        // Ensure at least one error card is expanded
        if (!_detectedErrors.Any(e => e.IsExpanded))
        {
            _detectedErrors[0].IsExpanded = true;
        }

        foreach (var err in _detectedErrors)
        {
            var card = BuildQuillBotCard(err);
            ErrorCardsContainer.Children.Add(card);
        }
    }

    private Border BuildQuillBotCard(QuillBotErrorItem err)
    {
        var card = new Border
        {
            Tag = err,
            CornerRadius = new CornerRadius(12),
            Background = Brushes.White,
            Cursor = err.IsExpanded ? Cursors.Arrow : Cursors.Hand
        };

        if (err.IsExpanded)
        {
            // Selected / Expanded Card Style (Screenshot 1, Card 1)
            card.BorderBrush = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#CBD5E1"));
            card.BorderThickness = new Thickness(1.2);
            card.Padding = new Thickness(16, 14, 16, 14);
            card.Margin = new Thickness(0, 0, 0, 12);
            card.Effect = new DropShadowEffect
            {
                Color = Colors.Black,
                BlurRadius = 10,
                ShadowDepth = 2,
                Opacity = 0.07
            };

            var grid = new Grid();
            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });

            // Left Red Pill Indicator
            var redPill = new Border
            {
                Width = 4,
                Height = 14,
                Background = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#E11D48")),
                CornerRadius = new CornerRadius(2),
                VerticalAlignment = VerticalAlignment.Top,
                Margin = new Thickness(0, 3, 10, 0)
            };
            Grid.SetColumn(redPill, 0);
            grid.Children.Add(redPill);

            var contentStack = new StackPanel();
            Grid.SetColumn(contentStack, 1);

            // Title Row: Action Description + Info Icon (Screenshot 1)
            var titleGrid = new Grid();
            var titleText = new TextBlock
            {
                Text = err.ActionDescription,
                FontSize = 12.5,
                FontWeight = FontWeights.SemiBold,
                Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#475569")),
                VerticalAlignment = VerticalAlignment.Center
            };
            var infoIcon = new TextBlock
            {
                Text = "ⓘ",
                FontSize = 12,
                Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#94A3B8")),
                HorizontalAlignment = HorizontalAlignment.Right,
                VerticalAlignment = VerticalAlignment.Center
            };
            titleGrid.Children.Add(titleText);
            titleGrid.Children.Add(infoIcon);
            contentStack.Children.Add(titleGrid);

            // Suggested Replacement in Bold Green (Screenshot 1)
            var replacementText = new TextBlock
            {
                Text = err.Replacement,
                FontSize = 17,
                FontWeight = FontWeights.Bold,
                Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#15803D")),
                Margin = new Thickness(0, 8, 0, 12)
            };
            contentStack.Children.Add(replacementText);

            // Action Buttons Row: [ Accept ]   Ignore ⊘ (Screenshot 1)
            var buttonsPanel = new StackPanel { Orientation = Orientation.Horizontal };

            var acceptBtn = new Button { Content = "Accept" };
            if (TryFindResource("QuillBotAcceptButton") is Style acceptStyle)
                acceptBtn.Style = acceptStyle;
            acceptBtn.Click += (s, e) => ApplyCorrection(err);
            buttonsPanel.Children.Add(acceptBtn);

            var ignoreBtn = new Button
            {
                Content = "Ignore ⊘",
                Margin = new Thickness(12, 0, 0, 0)
            };
            if (TryFindResource("QuillBotIgnoreButton") is Style ignoreStyle)
                ignoreBtn.Style = ignoreStyle;
            ignoreBtn.Click += (s, e) => IgnoreCorrection(err);
            buttonsPanel.Children.Add(ignoreBtn);

            contentStack.Children.Add(buttonsPanel);
            grid.Children.Add(contentStack);
            card.Child = grid;
        }
        else
        {
            // Collapsed Card Style (Screenshot 1, Card 2 & 3)
            card.BorderBrush = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#E2E8F0"));
            card.BorderThickness = new Thickness(1);
            card.Padding = new Thickness(16, 12, 16, 12);
            card.Margin = new Thickness(0, 0, 0, 10);

            var grid = new Grid();
            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });

            // Left Red Pill Indicator
            var redPill = new Border
            {
                Width = 4,
                Height = 14,
                Background = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#E11D48")),
                CornerRadius = new CornerRadius(2),
                VerticalAlignment = VerticalAlignment.Top,
                Margin = new Thickness(0, 3, 10, 0)
            };
            Grid.SetColumn(redPill, 0);
            grid.Children.Add(redPill);

            var contentStack = new StackPanel();
            Grid.SetColumn(contentStack, 1);

            var titleText = new TextBlock
            {
                Text = err.ActionDescription,
                FontSize = 12.5,
                Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#475569"))
            };
            contentStack.Children.Add(titleText);

            var origText = new TextBlock
            {
                Text = err.Original,
                FontSize = 14.5,
                FontWeight = FontWeights.SemiBold,
                Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#0F172A")),
                Margin = new Thickness(0, 4, 0, 0)
            };
            contentStack.Children.Add(origText);

            grid.Children.Add(contentStack);
            card.Child = grid;

            // Clicking collapsed card expands it and focuses that word in editor
            card.MouseLeftButtonDown += (s, e) =>
            {
                foreach (var other in _detectedErrors) other.IsExpanded = false;
                err.IsExpanded = true;
                RefreshErrorDisplay();
                SelectErrorInEditor(err);
            };
        }

        return card;
    }

    private void SelectErrorInEditor(QuillBotErrorItem err)
    {
        if (err.Range != null)
        {
            PlaygroundRichTextBox.Selection.Select(err.Range.Start, err.Range.End);
            PlaygroundRichTextBox.Focus();
        }
    }

    private void AcceptActiveError()
    {
        var active = _detectedErrors.FirstOrDefault(e => e.IsExpanded) ?? _detectedErrors.FirstOrDefault();
        if (active != null)
        {
            ApplyCorrection(active);
        }
    }

    private void ApplyCorrection(QuillBotErrorItem err)
    {
        if (err.Range != null)
        {
            _isApplyingFix = true;
            PlaygroundRichTextBox.TextChanged -= OnPlaygroundTextChanged;
            try
            {
                // 1. Replace ONLY that specific word at that exact target position!
                err.Range.Text = err.Replacement;
                err.Range.ApplyPropertyValue(Inline.TextDecorationsProperty, null);
                err.Range.ApplyPropertyValue(TextElement.BackgroundProperty, Brushes.Transparent);
                err.Range.ApplyPropertyValue(TextElement.ForegroundProperty, new SolidColorBrush((Color)ColorConverter.ConvertFromString("#0F172A")));

                // 2. Remove error from tracked list
                _detectedErrors.Remove(err);

                // 3. Automatically expand the next error
                if (_detectedErrors.Count > 0)
                {
                    _detectedErrors[0].IsExpanded = true;
                }

                RefreshErrorDisplay();
                StatusStripText.Text = $"Corrected: '{err.Original}' → '{err.Replacement}'";
            }
            finally
            {
                PlaygroundRichTextBox.TextChanged += OnPlaygroundTextChanged;
                _isApplyingFix = false;
                _debounceTimer.Stop();
                _debounceTimer.Start();
            }
        }
    }

    private void IgnoreCorrection(QuillBotErrorItem err)
    {
        if (err.Range != null)
        {
            err.Range.ApplyPropertyValue(Inline.TextDecorationsProperty, null);
            err.Range.ApplyPropertyValue(TextElement.BackgroundProperty, Brushes.Transparent);
            _detectedErrors.Remove(err);

            if (_detectedErrors.Count > 0)
            {
                _detectedErrors[0].IsExpanded = true;
            }

            RefreshErrorDisplay();
            StatusStripText.Text = $"Ignored suggestion for '{err.Original}'";
        }
    }

    private void OnPlaygroundSelectionChanged(object sender, RoutedEventArgs e)
    {
        if (_isApplyingFix || _detectedErrors.Count == 0) return;

        TextPointer caret = PlaygroundRichTextBox.CaretPosition;
        if (caret == null) return;

        // If user places cursor on a red word, expand its card in the sidebar
        QuillBotErrorItem? matched = _detectedErrors.FirstOrDefault(err => err.Range != null && err.Range.Contains(caret));
        if (matched != null && !matched.IsExpanded)
        {
            foreach (var other in _detectedErrors) other.IsExpanded = false;
            matched.IsExpanded = true;
            RefreshErrorDisplay();
        }
    }

    private void OnPlaygroundPreviewKeyDown(object sender, KeyEventArgs e)
    {
        // QuillBot Tab shortcut: instantly accepts the active card fix
        if (e.Key == Key.Tab && _detectedErrors.Count > 0)
        {
            AcceptActiveError();
            e.Handled = true;
        }
    }

    private static TextRange? FindWordRangeInDocument(FlowDocument doc, string searchText, List<QuillBotErrorItem> existingErrors)
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
}
