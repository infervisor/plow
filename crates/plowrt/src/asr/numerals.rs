//! Spoken digit runs to numerals: callers read phone, account and case numbers digit by digit and
//! an audio-LM spells them out ("nine one eight seven ..."); written transcripts use numerals.

const DIGITS: [(&str, u8); 11] = [
    ("zero", b'0'),
    ("oh", b'0'),
    ("one", b'1'),
    ("two", b'2'),
    ("three", b'3'),
    ("four", b'4'),
    ("five", b'5'),
    ("six", b'6'),
    ("seven", b'7'),
    ("eight", b'8'),
    ("nine", b'9'),
];

fn digit(word: &str) -> Option<u8> {
    if let [d @ b'0'..=b'9'] = word.as_bytes() {
        return Some(*d);
    }
    DIGITS.iter().find(|(name, _)| name.eq_ignore_ascii_case(word)).map(|&(_, d)| d)
}

/// One digit word ("double five" counts as one): its span, digits, and whether it is a bare "oh".
struct Token {
    start: usize,
    end: usize,
    digits: Vec<u8>,
    oh: bool,
}

/// Rewrite runs of at least three spoken digits (words or single numerals, separated by spaces,
/// commas, periods or hyphens) as numerals: ten digits `918-734-1538`, seven `734-1538`, other
/// lengths plain. A run needs two digits other than "oh", and "oh" at either end of it stays a word,
/// so interjections are untouched.
pub fn spoken_digits_to_numerals(text: &str) -> String {
    let words: Vec<(usize, usize)> = {
        let mut out = Vec::new();
        let mut start = None;
        for (i, c) in text.char_indices().chain([(text.len(), ' ')]) {
            match (c.is_alphanumeric(), start) {
                (true, None) => start = Some(i),
                (false, Some(s)) => {
                    out.push((s, i));
                    start = None;
                }
                _ => {}
            }
        }
        out
    };
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < words.len() {
        let (start, end) = words[i];
        let word = &text[start..end];
        let repeat = match word.to_ascii_lowercase().as_str() {
            "double" => 2,
            "triple" => 3,
            _ => 1,
        };
        if repeat > 1 {
            if let Some(d) = words.get(i + 1).and_then(|&(s, e)| digit(&text[s..e])) {
                tokens.push(Some(Token { start, end: words[i + 1].1, digits: vec![d; repeat], oh: false }));
                i += 2;
                continue;
            }
        }
        tokens.push(digit(word).map(|d| Token { start, end, digits: vec![d], oh: word.eq_ignore_ascii_case("oh") }));
        i += 1;
    }

    let mut out = String::with_capacity(text.len());
    let mut copied = 0;
    let mut run: Vec<&Token> = Vec::new();
    let flush = |run: &mut Vec<&Token>, out: &mut String, copied: &mut usize| {
        while run.last().is_some_and(|t| t.oh) {
            run.pop();
        }
        let first = run.iter().position(|t| !t.oh).unwrap_or(run.len());
        let run = &run[first..];
        let digits: Vec<u8> = run.iter().flat_map(|t| t.digits.iter().copied()).collect();
        if digits.len() >= 3 && run.iter().filter(|t| !t.oh).count() >= 2 {
            let digits = String::from_utf8(digits).expect("ASCII digits");
            out.push_str(&text[*copied..run[0].start]);
            out.push_str(&match digits.len() {
                10 => format!("{}-{}-{}", &digits[..3], &digits[3..6], &digits[6..]),
                7 => format!("{}-{}", &digits[..3], &digits[3..]),
                _ => digits,
            });
            *copied = run[run.len() - 1].end;
        }
    };
    for token in &tokens {
        match token {
            Some(t) => {
                let joined = run.last().is_some_and(|prev| {
                    text[prev.end..t.start].chars().all(|c| c.is_whitespace() || matches!(c, ',' | '.' | '-'))
                });
                if !joined {
                    flush(&mut run, &mut out, &mut copied);
                    run.clear();
                }
                run.push(t);
            }
            None => {
                flush(&mut run, &mut out, &mut copied);
                run.clear();
            }
        }
    }
    flush(&mut run, &mut out, &mut copied);
    out.push_str(&text[copied..]);
    out
}

#[cfg(test)]
mod tests {
    use super::spoken_digits_to_numerals as f;

    #[test]
    fn phone_and_account_numbers_become_numerals() {
        assert_eq!(f("My number is nine one eight seven three four one five three eight."), "My number is 918-734-1538.");
        assert_eq!(f("Eight one seven, seven six nine, zero zero six four"), "817-769-0064");
        assert_eq!(f("It's seven three four, double five, one two."), "It's 734-5512.");
        assert_eq!(f("case two zero two four"), "case 2024");
        assert_eq!(f("8 1 7 seven six nine"), "817769");
        assert_eq!(f("oh eight one three five five five oh one seven five oh"), "oh 813-555-0175 oh");
    }

    #[test]
    fn ordinary_speech_and_interjections_stay_words() {
        for text in [
            "Oh. Oh. Oh. Okay.",
            "Oh oh oh",
            "The one I called about, two weeks ago.",
            "One or two of them.",
            "Seven eight.",
            "At one, if he could call back.",
            "",
            "मैं वो कॉल",
        ] {
            assert_eq!(f(text), text);
        }
    }
}
