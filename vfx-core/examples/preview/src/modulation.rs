//! Expressions on effect parameters: a text box under each slider, like
//! `sin(time / 4)` or `value + bass * 0.5`, evaluated every frame. An empty box leaves
//! the slider in charge; a valid expression replaces the slider's value (which it can
//! still read as `value`).

use egui::{Color32, RichText, Ui};

use std::collections::HashMap;

use vfx_core::{ParamValue, ParamValues};

/// Variables an expression can read, in the order `Sources` stores them.
pub const VARIABLES: &[(&str, &str)] = &[
    ("time", "seconds since start (pauses with Animate)"),
    ("vtime", "video position, seconds"),
    ("progress", "video position, 0–1"),
    ("level", "audio loudness, 0–1"),
    ("bass", "20–250 Hz, 0–1"),
    ("lowmid", "250–1000 Hz, 0–1"),
    ("highmid", "1–4 kHz, 0–1"),
    ("treble", "4 kHz and up, 0–1"),
    // Face tracking; positions are 0–1 across and down the frame.
    ("face", "1 while a face is found, else 0"),
    ("facex", "face centre, x"),
    ("facey", "face centre, y"),
    ("facesize", "face width as a share of the frame width"),
    ("eyelx", "left eye on screen, x"),
    ("eyely", "left eye on screen, y"),
    ("eyerx", "right eye on screen, x"),
    ("eyery", "right eye on screen, y"),
    ("nosex", "nose tip, x"),
    ("nosey", "nose tip, y"),
    ("mouthx", "mouth centre, x"),
    ("mouthy", "mouth centre, y"),
];

/// One frame's variable values, indexed like VARIABLES.
pub type Sources = [f32; 20];

/// The face-tracking variables, in VARIABLES order from `face`.
pub type Face = [f32; 12];

/// Face values before any face is found (or with tracking off): centred guesses, so
/// expressions using them stay sensible, with `face` at 0.
pub const FACE_NONE: Face = [
    0.0, 0.5, 0.5, 0.0, 0.45, 0.45, 0.55, 0.45, 0.5, 0.5, 0.5, 0.6,
];

/// Builds the variable values. Video and audio read 0 without a video.
pub fn sources(time: f64, video: Option<(f64, f64)>, audio: [f32; 5], face: Face) -> Sources {
    let (position, duration) = video.unwrap_or((0.0, 1.0));
    let mut values = [0.0; 20];
    values[0] = time as f32;
    values[1] = position as f32;
    values[2] = (position / duration).clamp(0.0, 1.0) as f32;
    values[3..8].copy_from_slice(&audio);
    values[8..].copy_from_slice(&face);
    values
}

/// What an expression box drives: one effect's float or integer parameter, or one
/// coordinate of a point parameter.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Target {
    /// Index into the catalog.
    pub effect: usize,
    pub param: &'static str,
    /// For points: 0 for x, 1 for y.
    pub axis: Option<usize>,
}

impl Target {
    pub fn new(effect: usize, param: &'static str) -> Self {
        Self {
            effect,
            param,
            axis: None,
        }
    }

    pub fn axis(effect: usize, param: &'static str, axis: usize) -> Self {
        Self {
            effect,
            param,
            axis: Some(axis),
        }
    }

    fn read(&self, values: &ParamValues) -> Option<f32> {
        match (values.get(self.param)?, self.axis) {
            (ParamValue::Float(v), None) => Some(v),
            (ParamValue::Int(v), None) => Some(v as f32),
            (ParamValue::Point(p), Some(axis)) => Some(p[axis]),
            _ => None,
        }
    }

    /// Writes `v`, rounded for integers; the parameter clamps it to its range.
    fn write(&self, values: &mut ParamValues, v: f32) {
        let value = match (values.get(self.param), self.axis) {
            (Some(ParamValue::Float(_)), None) => ParamValue::Float(v),
            (Some(ParamValue::Int(_)), None) => ParamValue::Int(v.round() as i32),
            (Some(ParamValue::Point(mut p)), Some(axis)) => {
                p[axis] = v;
                ParamValue::Point(p)
            }
            _ => return,
        };
        let _ = values.set(self.param, value);
    }
}

/// One parameter's expression box.
#[derive(Default)]
struct Slot {
    text: String,
    compiled: Option<Expr>,
    error: Option<String>,
    /// What it evaluated to last frame, shown under the box.
    last: Option<f32>,
}

impl Slot {
    fn recompile(&mut self) {
        self.last = None;
        (self.compiled, self.error) = if self.text.trim().is_empty() {
            (None, None)
        } else {
            match Expr::parse(&self.text) {
                Ok(e) => (Some(e), None),
                Err(e) => (None, Some(e)),
            }
        };
    }
}

/// Every expression box, for every effect: switching effects keeps what was typed.
#[derive(Default)]
pub struct Expressions {
    slots: HashMap<Target, Slot>,
}

impl Expressions {
    /// `base` (effect `effect`'s values) with every valid expression applied. `dt` is the
    /// time since the last frame, for `smooth()`.
    pub fn apply(
        &mut self,
        effect: usize,
        base: &ParamValues,
        sources: &Sources,
        dt: f32,
    ) -> ParamValues {
        let mut values = base.clone();
        for (target, slot) in self.slots.iter_mut().filter(|(t, _)| t.effect == effect) {
            let (Some(expr), Some(value)) = (&mut slot.compiled, target.read(&values)) else {
                continue;
            };
            let v = expr.eval(&Env { sources, value, dt });
            // NaN or infinity (say, 1/0) keeps the slider's value.
            slot.last = v.is_finite().then_some(v);
            if let Some(v) = slot.last {
                target.write(&mut values, v);
            }
        }
        values
    }

    /// Forgets `smooth()` history, so an export starts from the source values.
    pub fn reset(&mut self) {
        for expr in self.slots.values_mut().filter_map(|s| s.compiled.as_mut()) {
            expr.reset();
        }
    }

    #[cfg(test)]
    pub fn set(&mut self, target: Target, text: &str) {
        let slot = self.slots.entry(target).or_default();
        slot.text = text.into();
        slot.recompile();
    }
}

/// The expression box for `target`, drawn under its slider.
pub fn field(ui: &mut Ui, exprs: &mut Expressions, target: Target) {
    let slot = exprs.slots.entry(target).or_default();
    let edit = egui::TextEdit::singleline(&mut slot.text)
        .id_salt(target)
        .hint_text("expression, e.g. sin(time/4)")
        .font(egui::TextStyle::Monospace)
        .desired_width(f32::INFINITY);
    let edit = if slot.error.is_some() {
        edit.text_color(Color32::from_rgb(230, 120, 90))
    } else {
        edit
    };
    if ui.add(edit).changed() {
        slot.recompile();
    }
    if let Some(error) = &slot.error {
        ui.label(
            RichText::new(error)
                .small()
                .color(Color32::from_rgb(230, 120, 90)),
        );
    } else if slot.compiled.is_some() {
        let shown = slot
            .last
            .map_or("= (not a number; slider used)".into(), |v| {
                format!("= {v:.3}")
            });
        ui.label(RichText::new(shown).small().monospace().weak());
    }
    ui.add_space(4.0);
}

/// The live variable values and what expressions can use.
pub fn help(ui: &mut Ui, sources: &Sources) {
    egui::Grid::new("variables").num_columns(2).show(ui, |ui| {
        for ((name, about), value) in VARIABLES.iter().zip(sources) {
            ui.label(RichText::new(*name).monospace())
                .on_hover_text(*about);
            ui.label(RichText::new(format!("{value:.2}")).monospace().weak());
            ui.end_row();
        }
        ui.label(RichText::new("value").monospace());
        ui.label(RichText::new("the slider").weak());
        ui.end_row();
    });
    ui.label(
        RichText::new(
            "Operators: + - * / % ^, comparisons (< > <= >= == !=) give 1 or 0.\n\
             if(condition, then, otherwise) picks by condition (anything but 0 is true), \
             e.g. if(facesize < 0.2, 0.2, facesize * 2); combine conditions with \
             and(a, b), or(a, b), not(a).\n\
             Functions: sin cos tan asin acos atan atan2 abs sqrt exp log pow min max \
             clamp floor ceil round fract sign mix step smoothstep, and \
             smooth(x, amount) to ease changes (amount 0–1).\nConstants: pi, tau.",
        )
        .small()
        .weak(),
    );
}

// ---- The expression language ----

#[derive(Clone, Copy, PartialEq, Debug)]
enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Pow,
    Lt,
    Gt,
    Le,
    Ge,
    Eq,
    Ne,
}

#[derive(Debug)]
enum Expr {
    Num(f32),
    Var(usize),
    Value,
    Neg(Box<Expr>),
    Bin(Op, Box<Expr>, Box<Expr>),
    Call(fn(&[f32]) -> f32, Vec<Expr>),
    /// smooth(x, amount), with its running value.
    Smooth(Box<Expr>, Box<Expr>, Option<f32>),
    /// if(condition, then, otherwise). Only the chosen branch runs, so a `smooth()`
    /// in the other one waits rather than tracking.
    If(Box<Expr>, Box<Expr>, Box<Expr>),
}

/// Conditions are true when not 0 (NaN counts as false).
fn truthy(v: f32) -> bool {
    v != 0.0 && !v.is_nan()
}

fn flag(c: bool) -> f32 {
    if c { 1.0 } else { 0.0 }
}

struct Env<'a> {
    sources: &'a Sources,
    value: f32,
    dt: f32,
}

/// name, argument count, implementation.
type Function = (&'static str, usize, fn(&[f32]) -> f32);

const FUNCTIONS: &[Function] = &[
    ("sin", 1, |a| a[0].sin()),
    ("cos", 1, |a| a[0].cos()),
    ("tan", 1, |a| a[0].tan()),
    ("asin", 1, |a| a[0].asin()),
    ("acos", 1, |a| a[0].acos()),
    ("atan", 1, |a| a[0].atan()),
    ("atan2", 2, |a| a[0].atan2(a[1])),
    ("abs", 1, |a| a[0].abs()),
    ("sqrt", 1, |a| a[0].sqrt()),
    ("exp", 1, |a| a[0].exp()),
    ("log", 1, |a| a[0].ln()),
    ("pow", 2, |a| a[0].powf(a[1])),
    ("min", 2, |a| a[0].min(a[1])),
    ("max", 2, |a| a[0].max(a[1])),
    ("clamp", 3, |a| a[0].max(a[1]).min(a[2])),
    ("floor", 1, |a| a[0].floor()),
    ("ceil", 1, |a| a[0].ceil()),
    ("round", 1, |a| a[0].round()),
    ("fract", 1, |a| a[0] - a[0].floor()),
    ("sign", 1, |a| if a[0] == 0.0 { 0.0 } else { a[0].signum() }),
    ("mix", 3, |a| a[0] + (a[1] - a[0]) * a[2]),
    ("lerp", 3, |a| a[0] + (a[1] - a[0]) * a[2]),
    ("step", 2, |a| if a[1] < a[0] { 0.0 } else { 1.0 }),
    ("smoothstep", 3, |a| {
        let t = ((a[2] - a[0]) / (a[1] - a[0])).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }),
    ("and", 2, |a| flag(truthy(a[0]) && truthy(a[1]))),
    ("or", 2, |a| flag(truthy(a[0]) || truthy(a[1]))),
    ("not", 1, |a| flag(!truthy(a[0]))),
];

impl Expr {
    fn parse(text: &str) -> Result<Self, String> {
        let tokens = tokenize(text)?;
        let mut parser = Parser { tokens, pos: 0 };
        let expr = parser.comparison()?;
        match parser.tokens.get(parser.pos) {
            None => Ok(expr),
            Some(t) => Err(format!("unexpected {t}")),
        }
    }

    fn eval(&mut self, env: &Env) -> f32 {
        match self {
            Expr::Num(n) => *n,
            Expr::Var(i) => env.sources[*i],
            Expr::Value => env.value,
            Expr::Neg(e) => -e.eval(env),
            Expr::Bin(op, a, b) => {
                let (a, b) = (a.eval(env), b.eval(env));
                match op {
                    Op::Add => a + b,
                    Op::Sub => a - b,
                    Op::Mul => a * b,
                    Op::Div => a / b,
                    // Always positive, so `time % 4` counts 0..4 like a loop.
                    Op::Rem => a.rem_euclid(b),
                    Op::Pow => a.powf(b),
                    Op::Lt => flag(a < b),
                    Op::Gt => flag(a > b),
                    Op::Le => flag(a <= b),
                    Op::Ge => flag(a >= b),
                    Op::Eq => flag(a == b),
                    Op::Ne => flag(a != b),
                }
            }
            Expr::Call(f, args) => {
                let values: Vec<f32> = args.iter_mut().map(|a| a.eval(env)).collect();
                f(&values)
            }
            Expr::Smooth(x, amount, state) => {
                let x = x.eval(env);
                // Frame-rate independent one-pole smoothing, tuned at 60 fps.
                let keep = amount.eval(env).clamp(0.0, 0.999).powf(env.dt * 60.0);
                let v = match *state {
                    Some(prev) if prev.is_finite() => prev + (x - prev) * (1.0 - keep),
                    _ => x,
                };
                *state = Some(v);
                v
            }
            Expr::If(condition, then, otherwise) => {
                if truthy(condition.eval(env)) {
                    then.eval(env)
                } else {
                    otherwise.eval(env)
                }
            }
        }
    }

    fn reset(&mut self) {
        match self {
            Expr::Num(_) | Expr::Var(_) | Expr::Value => {}
            Expr::Neg(e) => e.reset(),
            Expr::Bin(_, a, b) => {
                a.reset();
                b.reset();
            }
            Expr::Call(_, args) => args.iter_mut().for_each(Expr::reset),
            Expr::Smooth(x, amount, state) => {
                *state = None;
                x.reset();
                amount.reset();
            }
            Expr::If(condition, then, otherwise) => {
                condition.reset();
                then.reset();
                otherwise.reset();
            }
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
enum Token {
    Num(f32),
    Ident(String),
    Op(Op),
    Open,
    Close,
    Comma,
}

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Token::Num(n) => write!(f, "number {n}"),
            Token::Ident(s) => write!(f, "'{s}'"),
            Token::Op(_) => write!(f, "operator"),
            Token::Open => write!(f, "'('"),
            Token::Close => write!(f, "')'"),
            Token::Comma => write!(f, "','"),
        }
    }
}

fn tokenize(text: &str) -> Result<Vec<Token>, String> {
    let chars: Vec<char> = text.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        let two = |op| {
            if next == Some('=') {
                (Token::Op(op), 2)
            } else {
                (Token::Op(op), 1)
            }
        };
        let (token, len) = match c {
            c if c.is_whitespace() => {
                i += 1;
                continue;
            }
            c if c.is_ascii_digit() || c == '.' => {
                let end = (i..chars.len())
                    .find(|&j| !(chars[j].is_ascii_digit() || chars[j] == '.'))
                    .unwrap_or(chars.len());
                let s: String = chars[i..end].iter().collect();
                let n = s.parse().map_err(|_| format!("bad number '{s}'"))?;
                (Token::Num(n), end - i)
            }
            c if c.is_alphabetic() || c == '_' => {
                let end = (i..chars.len())
                    .find(|&j| !(chars[j].is_alphanumeric() || chars[j] == '_'))
                    .unwrap_or(chars.len());
                (Token::Ident(chars[i..end].iter().collect()), end - i)
            }
            '+' => (Token::Op(Op::Add), 1),
            '-' => (Token::Op(Op::Sub), 1),
            '*' if next == Some('*') => (Token::Op(Op::Pow), 2),
            '*' => (Token::Op(Op::Mul), 1),
            '/' => (Token::Op(Op::Div), 1),
            '%' => (Token::Op(Op::Rem), 1),
            '^' => (Token::Op(Op::Pow), 1),
            '(' => (Token::Open, 1),
            ')' => (Token::Close, 1),
            ',' => (Token::Comma, 1),
            '<' if next == Some('=') => two(Op::Le),
            '<' => (Token::Op(Op::Lt), 1),
            '>' if next == Some('=') => two(Op::Ge),
            '>' => (Token::Op(Op::Gt), 1),
            '=' if next == Some('=') => (Token::Op(Op::Eq), 2),
            '!' if next == Some('=') => (Token::Op(Op::Ne), 2),
            c => return Err(format!("unexpected '{c}'")),
        };
        tokens.push(token);
        i += len;
    }
    Ok(tokens)
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn next(&mut self) -> Result<Token, String> {
        let t = self
            .tokens
            .get(self.pos)
            .cloned()
            .ok_or("unfinished expression")?;
        self.pos += 1;
        Ok(t)
    }

    fn expect(&mut self, want: Token) -> Result<(), String> {
        match self.next()? {
            t if t == want => Ok(()),
            t => Err(format!("expected {want}, found {t}")),
        }
    }

    /// Binary operators from `ops`, left to right, over `operand`.
    fn binary(
        &mut self,
        ops: &[Op],
        operand: fn(&mut Self) -> Result<Expr, String>,
    ) -> Result<Expr, String> {
        let mut left = operand(self)?;
        while let Some(Token::Op(op)) = self.peek().cloned()
            && ops.contains(&op)
        {
            self.pos += 1;
            left = Expr::Bin(op, Box::new(left), Box::new(operand(self)?));
        }
        Ok(left)
    }

    fn comparison(&mut self) -> Result<Expr, String> {
        use Op::*;
        self.binary(&[Lt, Gt, Le, Ge, Eq, Ne], Self::sum)
    }

    fn sum(&mut self) -> Result<Expr, String> {
        self.binary(&[Op::Add, Op::Sub], Self::product)
    }

    fn product(&mut self) -> Result<Expr, String> {
        self.binary(&[Op::Mul, Op::Div, Op::Rem], Self::unary)
    }

    fn unary(&mut self) -> Result<Expr, String> {
        match self.peek() {
            Some(Token::Op(Op::Sub)) => {
                self.pos += 1;
                Ok(Expr::Neg(Box::new(self.unary()?)))
            }
            Some(Token::Op(Op::Add)) => {
                self.pos += 1;
                self.unary()
            }
            _ => self.power(),
        }
    }

    /// `^` binds tighter than unary minus on its left and groups right to left.
    fn power(&mut self) -> Result<Expr, String> {
        let base = self.atom()?;
        if self.peek() == Some(&Token::Op(Op::Pow)) {
            self.pos += 1;
            return Ok(Expr::Bin(Op::Pow, Box::new(base), Box::new(self.unary()?)));
        }
        Ok(base)
    }

    fn atom(&mut self) -> Result<Expr, String> {
        match self.next()? {
            Token::Num(n) => Ok(Expr::Num(n)),
            Token::Open => {
                let e = self.comparison()?;
                self.expect(Token::Close)?;
                Ok(e)
            }
            Token::Ident(name) if self.peek() == Some(&Token::Open) => {
                self.pos += 1;
                let mut args = Vec::new();
                if self.peek() != Some(&Token::Close) {
                    loop {
                        args.push(self.comparison()?);
                        if self.peek() == Some(&Token::Comma) {
                            self.pos += 1;
                        } else {
                            break;
                        }
                    }
                }
                self.expect(Token::Close)?;
                call(&name, args)
            }
            Token::Ident(name) => match name.as_str() {
                "value" => Ok(Expr::Value),
                "pi" => Ok(Expr::Num(std::f32::consts::PI)),
                "tau" => Ok(Expr::Num(std::f32::consts::TAU)),
                // `t` as a short name for time.
                "t" => Ok(Expr::Var(0)),
                _ => VARIABLES
                    .iter()
                    .position(|(v, _)| *v == name)
                    .map(Expr::Var)
                    .ok_or(format!("unknown name '{name}'")),
            },
            t => Err(format!("unexpected {t}")),
        }
    }
}

fn call(name: &str, mut args: Vec<Expr>) -> Result<Expr, String> {
    let arity = |n: usize| {
        if args.len() == n {
            Ok(())
        } else {
            Err(format!(
                "{name}() takes {n} argument{}",
                if n == 1 { "" } else { "s" }
            ))
        }
    };
    if name == "smooth" {
        arity(2)?;
        let amount = args.pop().expect("two arguments");
        let x = args.pop().expect("two arguments");
        return Ok(Expr::Smooth(Box::new(x), Box::new(amount), None));
    }
    if name == "if" {
        arity(3)?;
        let otherwise = args.pop().expect("three arguments");
        let then = args.pop().expect("three arguments");
        let condition = args.pop().expect("three arguments");
        return Ok(Expr::If(
            Box::new(condition),
            Box::new(then),
            Box::new(otherwise),
        ));
    }
    let (_, n, f) = FUNCTIONS
        .iter()
        .find(|(f, _, _)| *f == name)
        .ok_or(format!("unknown function '{name}'"))?;
    arity(*n)?;
    Ok(Expr::Call(*f, args))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval(text: &str, sources: &Sources, value: f32) -> f32 {
        Expr::parse(text).unwrap().eval(&Env {
            sources,
            value,
            dt: 1.0 / 60.0,
        })
    }

    #[test]
    fn arithmetic_follows_the_usual_precedence() {
        let s = sources(8.0, None, [0.0; 5], FACE_NONE);
        assert_eq!(eval("1 + 2 * 3", &s, 0.0), 7.0);
        assert_eq!(eval("(1 + 2) * 3", &s, 0.0), 9.0);
        assert_eq!(eval("-2^2", &s, 0.0), -4.0);
        assert_eq!(eval("2^3^2", &s, 0.0), 512.0);
        assert_eq!(eval("-1 % 4", &s, 0.0), 3.0);
        assert_eq!(eval("time > 5", &s, 0.0), 1.0);
        assert_eq!(eval("sin(time/4)", &s, 0.0), 2.0f32.sin());
        assert_eq!(eval("clamp(value * 10, 0, 1)", &s, 0.5), 1.0);
        assert_eq!(
            eval(
                "mix(0, 10, progress)",
                &sources(0.0, Some((5.0, 10.0)), [0.0; 5], FACE_NONE),
                0.0
            ),
            5.0
        );
    }

    #[test]
    fn conditions_pick_a_branch() {
        let mut face = FACE_NONE;
        face[3] = 0.1; // facesize
        let small = sources(0.0, None, [0.0; 5], face);
        face[3] = 0.4;
        let big = sources(0.0, None, [0.0; 5], face);
        let e = "if(facesize < 0.2, 0.2, facesize * 2)";
        assert_eq!(eval(e, &small, 0.0), 0.2);
        assert_eq!(eval(e, &big, 0.0), 0.8);
        // Nested, and combined conditions.
        let nested = "if(facesize < 0.2, 1, if(facesize < 0.5, 2, 3))";
        assert_eq!(eval(nested, &big, 0.0), 2.0);
        assert_eq!(eval("and(facesize > 0.3, not(face))", &big, 0.0), 1.0);
        assert_eq!(eval("or(0, 0)", &big, 0.0), 0.0);
        assert!(Expr::parse("if(1, 2)").is_err());
    }

    #[test]
    fn mistakes_are_reported_not_evaluated() {
        for bad in [
            "sin(",
            "1 +",
            "foo",
            "sin(1, 2)",
            "nope(1)",
            "1 2",
            "3 $ 4",
            ")",
        ] {
            assert!(Expr::parse(bad).is_err(), "{bad} should fail");
        }
    }

    fn values(effect: usize) -> ParamValues {
        ParamValues::new((crate::catalog::entries()[effect].make)().params())
    }

    /// Expressions and values for the named effect.
    fn setup(name: &str) -> (usize, ParamValues) {
        let effect = find(name).expect("in the catalog");
        (effect, values(effect))
    }

    use crate::catalog::find;

    const COLOUR: &str = "Colour Adjust";
    const BLUR: &str = "Gaussian Blur";
    const VIGNETTE: &str = "Vignette";
    const SEGCAM: &str = "segcam Motion (Rust)";

    #[test]
    fn expressions_override_sliders_and_empty_ones_dont() {
        let mut exprs = Expressions::default();
        let (colour, base) = setup(COLOUR);
        let s = sources(0.0, None, [0.0, 0.8, 0.0, 0.0, 0.0], FACE_NONE);
        exprs.set(Target::new(colour, "brightness"), "bass - 0.3");
        exprs.set(Target::new(colour, "contrast"), "  ");
        let out = exprs.apply(colour, &base, &s, 1.0 / 60.0);
        assert!((out.float("brightness") - 0.5).abs() < 1e-6);
        assert_eq!(out.float("contrast"), base.float("contrast"));
        exprs.set(Target::new(colour, "brightness"), "1 / 0");
        let out = exprs.apply(colour, &base, &s, 1.0 / 60.0);
        assert_eq!(out.float("brightness"), base.float("brightness"));
        // Another effect's expressions leave this one alone.
        exprs.set(Target::new(setup(BLUR).0, "radius"), "99");
        let out = exprs.apply(colour, &base, &s, 1.0 / 60.0);
        assert_eq!(out.float("contrast"), base.float("contrast"));
    }

    #[test]
    fn points_take_an_axis_and_integers_round() {
        let mut exprs = Expressions::default();
        let s = sources(0.0, None, [0.0; 5], FACE_NONE);
        let (vignette, base) = setup(VIGNETTE);
        exprs.set(Target::axis(vignette, "center", 1), "0.25");
        let out = exprs.apply(vignette, &base, &s, 1.0 / 60.0);
        assert_eq!(out.point("center"), [0.5, 0.25]);
        let (segcam, base) = setup(SEGCAM);
        exprs.set(Target::new(segcam, "background"), "40.6");
        exprs.set(Target::new(segcam, "output"), "9");
        let out = exprs.apply(segcam, &base, &s, 1.0 / 60.0);
        assert_eq!(out.int("background"), 41);
        assert_eq!(out.int("output"), 3, "clamped to the last mode");
    }

    #[test]
    fn smooth_eases_and_resets() {
        let mut exprs = Expressions::default();
        let (blur, base) = setup(BLUR);
        exprs.set(Target::new(blur, "radius"), "smooth(bass * 10, 0.9)");
        let quiet = sources(0.0, None, [0.0; 5], FACE_NONE);
        let loud = sources(0.0, None, [1.0; 5], FACE_NONE);
        let mut radius = |s| exprs.apply(blur, &base, s, 1.0 / 60.0).float("radius");
        assert_eq!(radius(&quiet), 0.0);
        let eased = radius(&loud);
        assert!(eased > 0.5 && eased < 2.0, "{eased}");
        drop(radius);
        exprs.reset();
        assert_eq!(
            exprs.apply(blur, &base, &loud, 1.0 / 60.0).float("radius"),
            10.0
        );
    }
}
