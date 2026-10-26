using System;
using System.Collections.Generic;
using System.Linq;

namespace TyPoTaKer.Desktop.Engine;

public sealed class RuleEdit
{
    public string Original { get; set; } = string.Empty;
    public string Replacement { get; set; } = string.Empty;
    public string ActionDescription { get; set; } = "Correct the spelling";
    public string Category { get; set; } = "Grammar";
}

public static class GrammarRuleEngine
{
    private static readonly HashSet<string> CommonLowercaseWords = new(StringComparer.OrdinalIgnoreCase)
    {
        "never", "always", "gonna", "wanna", "gotta", "very", "because", "the", "a", "an",
        "this", "that", "these", "those", "there", "their", "they", "them", "my", "your",
        "his", "her", "our", "its", "it", "we", "you", "he", "she", "and", "but", "or",
        "nor", "so", "for", "yet", "if", "when", "where", "why", "how", "what", "who",
        "whom", "which", "is", "are", "was", "were", "am", "be", "been", "being", "have",
        "has", "had", "do", "does", "did", "will", "would", "shall", "should", "can",
        "could", "may", "might", "must", "to", "in", "on", "at", "by", "with", "from",
        "into", "about", "against", "between", "through", "during", "before", "after",
        "above", "below", "under", "up", "down", "off", "over", "again", "further", "then",
        "once", "here", "all", "any", "both", "each", "few", "more", "most", "other",
        "some", "such", "no", "not", "only", "own", "same", "too", "just", "now", "also",
        "as", "well", "like", "love", "hate", "make", "take", "get", "go", "come", "see",
        "know", "think", "say", "tell", "give", "find", "use", "feel", "try", "leave",
        "call", "good", "bad", "new", "old", "first", "last", "long", "great", "little",
        "right", "big", "high", "different", "small", "large", "next", "early", "young",
        "important", "public", "able", "really", "already", "still", "even", "much", "many",
        "ever", "quite", "almost", "enough", "either", "neither", "since", "until", "while"
    };

    private static readonly HashSet<string> ProperNouns = new(StringComparer.OrdinalIgnoreCase)
    {
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
        "January", "February", "March", "April", "May", "June", "July", "August", "September",
        "October", "November", "December", "English", "American", "British", "French", "German",
        "Spanish", "Chinese", "Japanese", "God", "Bible", "Christian", "Islam", "Jewish"
    };

    private static readonly Dictionary<string, (string Replacement, string Action)> InformalWords = new(StringComparer.OrdinalIgnoreCase)
    {
        { "gonna", ("going to", "Replace informal wording") },
        { "wanna", ("want to", "Replace informal wording") },
        { "gotta", ("got to", "Replace informal wording") },
        { "kinda", ("kind of", "Replace informal wording") },
        { "sorta", ("sort of", "Replace informal wording") },
        { "cuz", ("because", "Replace informal wording") },
        { "dunno", ("do not know", "Replace informal wording") }
    };

    private static readonly Dictionary<string, string> CommonTypos = new(StringComparer.OrdinalIgnoreCase)
    {
        { "teh", "the" },
        { "si", "is" },
        { "bussiness", "business" },
        { "technolagy", "technology" },
        { "efficency", "efficiency" },
        { "seperate", "separate" },
        { "definately", "definitely" },
        { "writting", "writing" },
        { "untill", "until" },
        { "recieve", "receive" },
        { "occured", "occurred" },
        { "truely", "truly" },
        { "tommorrow", "tomorrow" },
        { "accomodate", "accommodate" },
        { "refered", "referred" },
        { "alot", "a lot" }
    };

    public static List<RuleEdit> FindRuleEdits(string text)
    {
        var edits = new List<RuleEdit>();
        if (string.IsNullOrWhiteSpace(text)) return edits;

        var sentences = text.Split(new[] { '.', '!', '?', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);

        foreach (var rawSentence in sentences)
        {
            string s = rawSentence.Trim();
            if (s.Length == 0) continue;

            string[] tokens = s.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
            if (tokens.Length == 0) continue;

            for (int i = 0; i < tokens.Length; i++)
            {
                string raw = tokens[i];
                string clean = raw.Trim(".,!?:;\"'()[]{}<>-/\\".ToCharArray());
                if (string.IsNullOrEmpty(clean)) continue;

                // 1. Sentence-Initial Capitalization (Word 0)
                if (i == 0)
                {
                    if (char.IsLower(clean[0]))
                    {
                        string cap = char.ToUpperInvariant(clean[0]) + (clean.Length > 1 ? clean.Substring(1) : "");
                        edits.Add(new RuleEdit
                        {
                            Original = clean,
                            Replacement = cap,
                            ActionDescription = "Correct the capitalization",
                            Category = "Grammar"
                        });
                        continue;
                    }
                }
                else
                {
                    // 2. Mid-Sentence Unnecessary Capitalization (Word i > 0)
                    // If word is capitalized, not "I", not acronym, not proper noun, but IS in common lowercase words
                    if (char.IsUpper(clean[0]) && clean != "I")
                    {
                        bool isAllUpper = clean.Length <= 5 && clean.All(char.IsUpper);
                        if (!isAllUpper && !ProperNouns.Contains(clean) && CommonLowercaseWords.Contains(clean))
                        {
                            // If it's an informal word like Gonna, prefer "going to" or lowercase "gonna"
                            if (InformalWords.TryGetValue(clean, out var inf))
                            {
                                edits.Add(new RuleEdit
                                {
                                    Original = clean,
                                    Replacement = inf.Replacement,
                                    ActionDescription = inf.Action,
                                    Category = "Grammar"
                                });
                            }
                            else
                            {
                                edits.Add(new RuleEdit
                                {
                                    Original = clean,
                                    Replacement = clean.ToLowerInvariant(),
                                    ActionDescription = "Correct the capitalization",
                                    Category = "Grammar"
                                });
                            }
                            continue;
                        }
                    }
                }

                // 3. Informal Wording Check (e.g. gonna -> going to)
                if (InformalWords.TryGetValue(clean, out var informalFix))
                {
                    if (!edits.Any(e => e.Original.Equals(clean, StringComparison.OrdinalIgnoreCase)))
                    {
                        edits.Add(new RuleEdit
                        {
                            Original = clean,
                            Replacement = informalFix.Replacement,
                            ActionDescription = informalFix.Action,
                            Category = "Grammar"
                        });
                        continue;
                    }
                }

                // 4. Common Typo Check (e.g. teh -> the, si -> is, bussiness -> business)
                if (CommonTypos.TryGetValue(clean, out var typoFix))
                {
                    if (!edits.Any(e => e.Original.Equals(clean, StringComparison.OrdinalIgnoreCase)))
                    {
                        edits.Add(new RuleEdit
                        {
                            Original = clean,
                            Replacement = typoFix,
                            ActionDescription = "Correct the spelling",
                            Category = "Spelling"
                        });
                        continue;
                    }
                }
            }
        }

        return edits;
    }

    public static string ApplyEdits(string text, List<RuleEdit> edits)
    {
        if (string.IsNullOrEmpty(text) || edits == null || edits.Count == 0)
            return text;

        string result = text;
        foreach (var edit in edits)
        {
            if (string.IsNullOrEmpty(edit.Original)) continue;
            int idx = 0;
            while ((idx = result.IndexOf(edit.Original, idx, StringComparison.OrdinalIgnoreCase)) >= 0)
            {
                bool boundBefore = (idx == 0) || !char.IsLetterOrDigit(result[idx - 1]);
                bool boundAfter = (idx + edit.Original.Length >= result.Length) || !char.IsLetterOrDigit(result[idx + edit.Original.Length]);
                if (boundBefore && boundAfter)
                {
                    result = result.Substring(0, idx) + edit.Replacement + result.Substring(idx + edit.Original.Length);
                    idx += edit.Replacement.Length;
                }
                else
                {
                    idx += edit.Original.Length;
                }
            }
        }
        return result;
    }
}
