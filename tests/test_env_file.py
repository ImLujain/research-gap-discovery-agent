import os
from research_gap_agent.config import Settings, load_env_file


def test_env_file_parsing_and_precedence(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nexport ANTHROPIC_API_KEY="sk-test"\nRGA_MAILTO=\'me@example.org\'\nS2_API_KEY=\n\nBAD LINE\nRGA_MODEL=claude-opus-5-5\n')
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("S2_API_KEY", raising=False)
    monkeypatch.setenv("RGA_MODEL", "already-set")
    monkeypatch.setenv("RGA_MAILTO", "shell@example.org")
    loaded = load_env_file(env)
    assert loaded["ANTHROPIC_API_KEY"] == "sk-test"
    assert "RGA_MAILTO" not in loaded and Settings().mailto == "shell@example.org"  # shell value wins
    assert Settings().model == "already-set"
    assert loaded["S2_API_KEY"] == "" and Settings().s2_api_key == ""


def test_env_file_found_in_cwd(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("NCBI_API_KEY=abc\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    assert load_env_file() == {"NCBI_API_KEY": "abc"}
    assert Settings().ncbi_api_key == "abc"


def test_missing_env_file_is_fine(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_env_file(tmp_path / "nope.env") == {}


def test_rga_env_file_override_pins_the_file(tmp_path, monkeypatch):
    from research_gap_agent.config import default_env_path

    pinned = tmp_path / "pinned.env"
    pinned.write_text("SERPAPI_KEY=pinned\n")
    (tmp_path / ".env").write_text("SERPAPI_KEY=cwd\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RGA_ENV_FILE", str(pinned))
    monkeypatch.delenv("SERPAPI_KEY", raising=False)
    assert load_env_file() == {"SERPAPI_KEY": "pinned"} and default_env_path() == pinned


def test_update_env_file_replaces_keeps_comments_and_removes(tmp_path, monkeypatch):
    from research_gap_agent.config import update_env_file

    env = tmp_path / ".env"
    env.write_text("# keep me\nANTHROPIC_API_KEY=old\nexport S2_API_KEY=s2\nOTHER=x\n")
    monkeypatch.delenv("SERPAPI_KEY", raising=False)
    update_env_file({"ANTHROPIC_API_KEY": "new", "SERPAPI_KEY": "serp", "S2_API_KEY": ""}, env)
    text = env.read_text()
    assert text == "# keep me\nANTHROPIC_API_KEY=new\nOTHER=x\nSERPAPI_KEY=serp\n"
    assert os.environ["ANTHROPIC_API_KEY"] == "new" and os.environ["SERPAPI_KEY"] == "serp"
    assert "S2_API_KEY" not in os.environ
    assert oct(env.stat().st_mode & 0o777) == "0o600"


def test_update_env_file_creates_file(tmp_path):
    from research_gap_agent.config import update_env_file

    path = update_env_file({"RGA_MAILTO": "me@example.org"}, tmp_path / "sub" / ".env")
    assert path.read_text() == "RGA_MAILTO=me@example.org\n"
