//! Drawing the three panes: results, preview, and the path/why strip, plus the help overlay.

use super::app::{App, Areas, Metrics};
use super::preview::{Preview, why};
use crate::paths::tilde;
use crate::search::Hit;
use crate::store::Kind;
use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Clear, List, ListItem, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Wrap,
};
use std::path::Path;
use std::time::Duration;

const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const HINTS: &str = " ↑↓ move  ⇧↑↓ scroll  ⏎ open  ^o reveal  ^y copy  tab kind  F1 help  esc quit ";
const NOTICE_FOR: Duration = Duration::from_secs(2);

const HELP: [(&str, &str); 12] = [
    ("type", "search as you type (\"code:\" first searches code)"),
    ("↑ ↓  ^p ^n", "move through results"),
    ("⇧↑ ⇧↓  PgUp PgDn", "scroll the preview"),
    ("mouse", "click a result, click again to open; wheel scrolls"),
    ("⏎", "open the file ($EDITOR for code)"),
    ("^v", "PDFs: switch between the page's text and a picture of it"),
    ("^o", "reveal in the file manager"),
    ("^y", "copy the path (works over ssh)"),
    ("tab", "show all, docs or code"),
    ("^u  ^w", "clear the query, delete a word"),
    ("? or F1", "this help (? only on an empty query)"),
    ("esc  ^c", "quit"),
];

/// Draw the whole interface.
pub fn render(frame: &mut Frame, app: &App) {
    let area = frame.area();
    let spinner = if app.searching() { format!(" {}", SPINNER[app.tick % SPINNER.len()]) } else { String::new() };
    let mut title = vec![
        Span::raw(" rummage ─ "),
        Span::styled(app.query.clone(), Style::default().add_modifier(Modifier::BOLD)),
        Span::styled("█", Style::default().fg(Color::Gray)),
    ];
    title.push(Span::raw(format!("{spinner} ")));
    let footer = match &app.notice {
        Some((message, at)) if at.elapsed() < NOTICE_FOR => format!(" {message} "),
        _ => HINTS.to_string(),
    };
    let mut block = Block::bordered().title(Line::from(title)).title_bottom(Line::from(footer));
    if app.filter != super::app::Filter::All {
        block = block.title(Line::from(format!(" {} only ", app.filter.label())).right_aligned());
    }
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let [left, right] = Layout::horizontal([Constraint::Percentage(45), Constraint::Percentage(55)]).areas(inner);
    render_results(frame, app, left);
    render_right(frame, app, right);
    if app.help {
        render_help(frame, area);
    }
}

fn render_results(frame: &mut Frame, app: &App, area: Rect) {
    app.areas.set(Areas { results: area, ..app.areas.get() });
    let visible = app.visible();
    if visible.is_empty() {
        let message = if let Some(error) = &app.error {
            format!("search failed: {error}")
        } else if app.query.trim().is_empty() {
            "type to search".to_string()
        } else if app.searching() {
            format!("{} searching…", SPINNER[app.tick % SPINNER.len()])
        } else {
            "no good matches".to_string()
        };
        let style =
            if app.error.is_some() { Style::default().fg(Color::Red) } else { Style::default().fg(Color::DarkGray) };
        frame.render_widget(Paragraph::new(message).style(style).wrap(Wrap { trim: true }), area);
        return;
    }
    let width = area.width as usize;
    let items: Vec<ListItem> = visible
        .iter()
        .enumerate()
        .map(|(row, &i)| ListItem::new(row_line(&app.hits[i], row == app.selected, width)))
        .collect();
    let mut state = app.list.borrow_mut();
    state.select(Some(app.selected));
    let list = List::new(items).highlight_style(Style::default().add_modifier(Modifier::REVERSED | Modifier::BOLD));
    frame.render_stateful_widget(list, area, &mut state);
}

/// One result row: marker, kind, file name, and the passage number at the right edge.
fn row_line(hit: &Hit, selected: bool, width: usize) -> Line<'static> {
    let name = Path::new(&hit.path).file_name().map_or(hit.path.clone(), |n| n.to_string_lossy().into_owned());
    let place = match (hit.kind, hit.loc) {
        (Kind::Pdf, Some(loc)) => format!("p{}", loc.round() as i64),
        (_, Some(loc)) if loc > 1.0 => format!("#{}", loc.round() as i64),
        _ => String::new(),
    };
    let marker = if selected { "▸" } else { " " };
    let kind_color = match hit.kind {
        Kind::Doc => Color::Cyan,
        Kind::Pdf => Color::Magenta,
        Kind::Code => Color::Yellow,
    };
    let fixed = 2 + 5 + place.chars().count() + 1;
    let room = width.saturating_sub(fixed).max(1);
    let name = truncate(&name, room);
    let pad = room.saturating_sub(name.chars().count());
    let kind_style = if selected { Style::default() } else { Style::default().fg(kind_color) };
    Line::from(vec![
        Span::raw(format!("{marker} ")),
        Span::styled(format!("{:<5}", hit.kind.as_str()), kind_style),
        Span::raw(name),
        Span::raw(" ".repeat(pad + 1)),
        Span::raw(place),
    ])
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let keep: String = s.chars().take(max.saturating_sub(1)).collect();
    format!("{keep}…")
}

fn render_right(frame: &mut Frame, app: &App, area: Rect) {
    let [preview_area, info_area] = Layout::vertical([Constraint::Min(1), Constraint::Length(3)]).areas(area);
    let divider =
        Block::bordered().borders(ratatui::widgets::Borders::LEFT).border_style(Style::default().fg(Color::DarkGray));
    let preview_area = {
        let inner = divider.inner(preview_area);
        frame.render_widget(divider.clone(), preview_area);
        inner
    };
    app.areas.set(Areas { preview: preview_area, ..app.areas.get() });
    let info_block = divider.clone();
    let info_inner = info_block.inner(info_area);
    frame.render_widget(info_block, info_area);

    let (Some(i), Some(preview)) = (app.selected_hit(), app.preview()) else { return };
    let hit = &app.hits[i];
    let page_view = app.page_view && hit.kind == Kind::Pdf;
    if page_view {
        match app.page_image.borrow_mut().as_mut() {
            Some((_, Ok(protocol))) => {
                frame.render_stateful_widget(ratatui_image::StatefulImage::default(), preview_area, protocol)
            }
            Some((_, Err(message))) => frame.render_widget(
                Paragraph::new(message.clone()).style(Style::default().fg(Color::Red)).wrap(Wrap { trim: true }),
                preview_area,
            ),
            None => {}
        }
    }
    match preview {
        _ if page_view => {}
        Preview::Text(shown) | Preview::Code(shown) => {
            let paragraph = Paragraph::new(shown.lines.clone()).wrap(Wrap { trim: false });
            // Counting wrapped rows means laying out every line: done once per preview and width.
            let metrics = match app.metrics.get() {
                Some(m) if m.width == preview_area.width => m,
                _ => {
                    let landing = shown.anchor.unwrap_or(shown.focus);
                    let before = Paragraph::new(shown.lines[..landing.min(shown.lines.len())].to_vec())
                        .wrap(Wrap { trim: false });
                    let m = Metrics {
                        width: preview_area.width,
                        landing: before.line_count(preview_area.width),
                        total: paragraph.line_count(preview_area.width),
                    };
                    app.metrics.set(Some(m));
                    m
                }
            };
            let height = preview_area.height as usize;
            let max = metrics.total.saturating_sub(height);
            // Open with the first marked word a third of the way down, measured at the pane's size now (a later resize
            // may move it out of view; it isn't chased). Nothing marked: open at the passage, a line of context above.
            let lead = if shown.anchor.is_some() { height / 3 } else { 1 };
            let scroll = app.scroll.get().unwrap_or(metrics.landing.saturating_sub(lead)).min(max);
            app.scroll.set(Some(scroll));
            frame.render_widget(paragraph.scroll((scroll.min(u16::MAX as usize) as u16, 0)), preview_area);
            if metrics.total > height {
                let mut bar = ScrollbarState::new(max).position(scroll).viewport_content_length(height);
                let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight).begin_symbol(None).end_symbol(None);
                frame.render_stateful_widget(scrollbar, preview_area, &mut bar);
            }
        }
        Preview::Message(message) => {
            frame.render_widget(
                Paragraph::new(message.clone()).style(Style::default().fg(Color::DarkGray)),
                preview_area,
            );
        }
    }
    let mut detail = why(hit);
    if hit.kind == Kind::Pdf {
        detail += if page_view { " · ^v text" } else { " · ^v page" };
    }
    let info = vec![
        Line::from(Span::styled(tilde(Path::new(&hit.path)), Style::default().add_modifier(Modifier::BOLD))),
        Line::from(Span::styled(detail, Style::default().fg(Color::DarkGray))),
    ];
    frame.render_widget(Paragraph::new(info), info_inner);
}

fn render_help(frame: &mut Frame, area: Rect) {
    let width = 72.min(area.width);
    let height = (HELP.len() as u16 + 4).min(area.height);
    let popup = Rect::new(area.x + (area.width - width) / 2, area.y + (area.height - height) / 2, width, height);
    let mut lines: Vec<Line> = HELP
        .iter()
        .map(|(keys, what)| {
            Line::from(vec![
                Span::styled(format!("{keys:<17}"), Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(*what),
            ])
        })
        .collect();
    lines.push(Line::from(""));
    lines.push(Line::styled("any key closes this", Style::default().fg(Color::DarkGray)));
    let block = Block::bordered().title(" help ").title_alignment(Alignment::Center);
    frame.render_widget(Clear, popup);
    frame.render_widget(Paragraph::new(lines).block(block), popup);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::app::tests::hit;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use std::sync::mpsc;

    fn draw(app: &mut App, width: u16, height: u16) -> String {
        app.refresh_preview();
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal.draw(|f| render(f, app)).unwrap();
        let buffer = terminal.backend().buffer().clone();
        (0..height)
            .map(|y| (0..width).map(|x| buffer[(x, y)].symbol().to_string()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn new_app() -> App {
        App::new(mpsc::channel().0)
    }

    fn with_hits() -> App {
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("rummage-tui-test-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let note = dir.join("beach-trip.md");
        std::fs::write(&note, "# Beach trip\nWe took the Dog to the beach on Saturday.\nIt rained.\n").unwrap();
        let mut app = new_app();
        app.query = "dog on the beach".into();
        let mut first = hit(note.to_str().unwrap(), Kind::Doc);
        first.loc = Some(1.0);
        let mut second = hit("/nowhere/lease.txt", Kind::Doc);
        second.loc = Some(3.0);
        app.hits = vec![first, second, hit("/nowhere/retry.rs", Kind::Code)];
        app
    }

    #[test]
    fn empty_query() {
        let screen = draw(&mut new_app(), 80, 12);
        assert!(screen.contains("rummage ─ █"), "{screen}");
        assert!(screen.contains("type to search"));
        assert!(screen.contains("esc quit"));
    }

    #[test]
    fn results_and_preview() {
        let screen = draw(&mut with_hits(), 140, 14);
        assert!(screen.contains("rummage ─ dog on the beach█"), "{screen}");
        assert!(screen.contains("▸ doc  beach-trip.md"), "{screen}");
        assert!(screen.contains("lease.txt") && screen.contains("#3"));
        assert!(screen.contains("code retry.rs"));
        assert!(screen.contains("We took the Dog to the beach"), "{screen}");
        assert!(screen.contains("beach-trip.md") && screen.contains("meaning z 3.1 · words #2"));
    }

    #[test]
    fn matched_words_are_highlighted() {
        let mut app = with_hits();
        app.refresh_preview();
        let Some(Preview::Text(shown)) = app.preview() else { panic!("no text preview") };
        let marked: Vec<&str> = shown.lines[1]
            .spans
            .iter()
            .filter(|s| s.style.bg == Some(Color::Yellow))
            .map(|s| s.content.as_ref())
            .collect();
        assert_eq!(marked, vec!["Dog", "beach"]);
    }

    #[test]
    fn a_narrow_pane_opens_with_the_match_in_view() {
        let dir = std::env::temp_dir().join(format!("rummage-tui-narrow-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let note = dir.join("play.txt");
        // One passage: a long run of prose (dozens of wrapped rows in a narrow pane), then the matched word.
        let filler = "and so the evening wore on with talk of nothing much at all ".repeat(25);
        std::fs::write(&note, format!("ACT ONE\n{filler}\nThe soldiers were fighting in the square.\nEXIT\n")).unwrap();
        let mut app = new_app();
        app.query = "fighting".into();
        let mut found = hit(note.to_str().unwrap(), Kind::Doc);
        found.loc = Some(1.0);
        app.hits = vec![found];
        let screen = draw(&mut app, 70, 16);
        assert!(screen.contains("fighting"), "{screen}");
        assert!(!screen.contains("ACT ONE"), "scrolled past the passage start: {screen}");
        // Meaning-only (nothing marked): the pane opens at the passage start.
        app.query = "a quarrel".into();
        let screen = draw(&mut app, 70, 16);
        assert!(screen.contains("ACT ONE"), "{screen}");
    }

    #[test]
    fn missing_file_says_so() {
        let mut app = with_hits();
        app.selected = 1;
        let screen = draw(&mut app, 100, 14);
        assert!(screen.contains("can't read file"), "{screen}");
        assert!(screen.contains("/nowhere/lease.txt"));
    }

    #[test]
    fn help_overlay() {
        let mut app = new_app();
        app.help = true;
        let screen = draw(&mut app, 90, 20);
        assert!(screen.contains("help") && screen.contains("reveal in the file manager"), "{screen}");
        assert!(screen.contains("any key closes this"));
    }

    #[test]
    fn no_matches_and_filter() {
        let mut app = new_app();
        app.query = "zzzz".into();
        let screen = draw(&mut app, 80, 12);
        assert!(screen.contains("no good matches"), "{screen}");
        let mut app = with_hits();
        app.filter = crate::tui::app::Filter::Only(Kind::Code);
        let screen = draw(&mut app, 100, 14);
        assert!(screen.contains("code only") && !screen.contains("lease.txt"), "{screen}");
    }
}
