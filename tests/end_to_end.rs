//! The whole thing, through the `rummage` binary on a temporary folder: indexing, re-syncing, long files in two
//! passes, the background service (start, grep, stop, recovery after a crash). Needs the model files; skipped
//! without them, as on CI.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn model_present() -> bool {
    if rummage::paths::ModelFiles::locate().is_ok() {
        return true;
    }
    eprintln!("skipped: no model files");
    false
}

/// A temporary folder of files and an index of its own; stops that index's service when dropped.
struct Sandbox {
    root: PathBuf,
}

impl Sandbox {
    fn new(name: &str) -> Sandbox {
        let root = std::env::temp_dir().join(format!("rummage-e2e-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("files")).unwrap();
        Sandbox { root }
    }

    fn files(&self) -> PathBuf {
        self.root.join("files")
    }

    fn write(&self, name: &str, text: &str) {
        std::fs::write(self.files().join(name), text).unwrap();
    }

    fn rummage(&self, args: &[&str]) -> Output {
        let output = Command::new(env!("CARGO_BIN_EXE_rummage"))
            .arg("--index")
            .arg(self.root.join("index"))
            .args(args)
            .output()
            .unwrap();
        assert!(output.status.success(), "rummage {args:?}: {}", String::from_utf8_lossy(&output.stderr));
        output
    }

    fn index(&self) -> String {
        String::from_utf8_lossy(&self.rummage(&["index", self.files().to_str().unwrap()]).stderr).into_owned()
    }

    /// File names of the hits, best first, with each hit's JSON.
    fn search(&self, query: &str, extra: &[&str]) -> Vec<(String, serde_json::Value)> {
        let mut args = vec!["search", query, "--json"];
        args.extend_from_slice(extra);
        let hits: Vec<serde_json::Value> = serde_json::from_slice(&self.rummage(&args).stdout).unwrap();
        hits.into_iter()
            .map(|h| (Path::new(h["path"].as_str().unwrap()).file_name().unwrap().to_string_lossy().into_owned(), h))
            .collect()
    }

    fn names(&self, query: &str, extra: &[&str]) -> Vec<String> {
        self.search(query, extra).into_iter().map(|(name, _)| name).collect()
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        let _ = Command::new(env!("CARGO_BIN_EXE_rummage"))
            .arg("--index")
            .arg(self.root.join("index"))
            .arg("stop")
            .output();
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

#[test]
fn index_search_and_resync() {
    if !model_present() {
        return;
    }
    let s = Sandbox::new("resync");
    s.write("beach-trip.md", "We took the puppy to the seaside. She ran along the sand chasing the waves for an hour.");
    s.write("lease.txt", "Clause 9: Pets. The tenant may keep one cat or one small dog with written consent.");
    s.write("recipe.md", "Brown bread: wholemeal flour, bread soda, salt and buttermilk; bake 45 minutes.");
    s.write("invoice.md", "Invoice INV20931 from Acme Hosting, due 30 November, for server rental.");
    // Unrelated notes: with only a handful of passages nothing can stand out from the rest.
    s.write("sleep.md", "Podcast notes: keep the bedroom cool, no caffeine after 2pm, and wake at a fixed time.");
    s.write("standup.md", "Standup: the payment page throws card declined errors; check the webhook retries.");
    s.write("travel.md", "Booked flights to Lisbon for the conference, leaving Dublin on the 14th.");
    s.write("garden.md", "Planted tomatoes and basil along the south wall; water every second evening.");
    s.write("id_rsa", "-----BEGIN OPENSSH PRIVATE KEY----- not to be read");
    assert!(s.index().contains("8 files"), "the key is never read");

    // Meaning, words, and an exact lookup.
    assert_eq!(
        s.names("a dog playing on the beach", &["--no-serve"]).first().map(String::as_str),
        Some("beach-trip.md")
    );
    assert_eq!(s.names("can I keep a pet in the flat", &["--no-serve"]).first().map(String::as_str), Some("lease.txt"));
    assert_eq!(s.names("INV20931", &["--no-serve"]).first().map(String::as_str), Some("invoice.md"));

    // Change one file, delete one, add one: only those are redone.
    std::thread::sleep(std::time::Duration::from_millis(1100)); // a new mtime, at one-second precision
    s.write("lease.txt", "Clause 9: Pets. A parrot is allowed in the flat, but no dogs.");
    std::fs::remove_file(s.files().join("recipe.md")).unwrap();
    s.write("mortgage.md", "Meeting with the bank about renewing the mortgage next Tuesday.");
    let report = s.index();
    assert!(report.contains("2 new or changed, 1 gone"), "{report}");
    assert_eq!(s.names("parrot", &["--no-serve"]).first().map(String::as_str), Some("lease.txt"));
    assert_eq!(s.names("home loan", &["--no-serve"]).first().map(String::as_str), Some("mortgage.md"));
    assert!(!s.names("brown bread recipe", &["--no-serve"]).contains(&"recipe.md".to_string()));
    assert!(s.index().contains("0 new or changed, 0 gone"), "a second run with nothing changed does nothing");
}

#[test]
fn long_documents_go_in_two_passes() {
    if !model_present() {
        return;
    }
    let s = Sandbox::new("long");
    // About 110 passages: past the first pass's 40. The phrase sits near the end.
    let filler: String = (0..900)
        .map(|i| format!("Entry {i}: the committee reviewed the quarterly figures and adjourned without changes. "))
        .collect();
    s.write("minutes.txt", &format!("{filler}\n\nThe lighthouse keeper's cat was named Admiral Whiskers.\n\n{filler}"));
    let report = s.index();
    assert!(report.contains("reading the rest of long files"), "{report}");
    let hits = s.search("Admiral Whiskers the lighthouse cat", &["--no-serve"]);
    let (name, hit) = hits.first().expect("found");
    assert_eq!(name, "minutes.txt");
    assert!(hit["loc"].as_f64().unwrap() > 40.0, "found in a later pass: {hit}");
}

#[test]
fn service_greps_stops_and_recovers() {
    if !model_present() {
        return;
    }
    let s = Sandbox::new("service");
    s.write("standup.md", "Payment page throws card declined for some users; check the webhook retries.");
    // A .log file isn't indexed, only grepped.
    s.write(
        "server.log",
        "12:00 INFO ok\n12:01 ERROR checkout failed: card declined by issuer (code 05)\n12:02 INFO ok\n",
    );
    s.index();

    // Through the service (started by the search): the log is found by grep, at its line.
    let hits = s.search("card declined by issuer", &[]);
    let log = hits.iter().find(|(name, _)| name == "server.log").expect("grep found the log");
    assert_eq!((log.1["grep"].as_bool(), log.1["line"].as_u64()), (Some(true), Some(2)));
    assert_eq!(s.names("/code 0[0-9]/", &[]), vec!["server.log"], "a regex goes to grep");
    assert!(s.search("card declined by issuer", &["--no-grep"]).iter().all(|(name, _)| name != "server.log"));

    // Stop, then a search starts it again.
    assert!(String::from_utf8_lossy(&s.rummage(&["stop"]).stderr).contains("stopped"));
    assert!(!s.names("card declined", &[]).is_empty());

    // A service killed outright leaves its socket behind: the next search clears it and starts a new one.
    s.rummage(&["stop"]);
    let mut service = Command::new(env!("CARGO_BIN_EXE_rummage"))
        .arg("--index")
        .arg(s.root.join("index"))
        .args(["serve", "--linger", "0"])
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let socket = rummage::serve::socket_path(&s.root.join("index"));
    for _ in 0..200 {
        if socket.exists() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert!(socket.exists(), "the service came up");
    service.kill().unwrap();
    service.wait().unwrap();
    assert!(socket.exists(), "killed outright, it left its socket");
    assert!(!s.names("card declined", &[]).is_empty(), "recovered");
}
