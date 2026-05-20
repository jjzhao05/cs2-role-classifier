import sys
import time
import requests
import mwparserfromhell
import pandas as pd

BASE = "https://liquipedia.net/counterstrike/api.php"

# Liquipedia requires a descriptive User-Agent; fill in your contact info.
HEADERS = {
    "User-Agent": "cs2-role-classifier/0.1 (your_email@example.com)",
    "Accept-Encoding": "gzip",
}

REQUEST_DELAY = 5   # seconds between requests (Liquipedia rate limit)
INPUT_CSV  = sys.argv[1] if len(sys.argv) > 1 else "output.csv"
OUTPUT_CSV = "liquipedia_player_roles.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_players(csv_path: str) -> list[str]:
    df = pd.read_csv(csv_path)
    return sorted(df.iloc[:, 0].dropna().astype(str).unique().tolist())


def fetch_wikitext(session: requests.Session, page: str) -> str | None:
    params = {
        "action": "query",
        "prop": "revisions",
        "titles": page,
        "rvprop": "content",
        "format": "json",
        "formatversion": "2",
    }
    r = session.get(BASE, params=params, timeout=20)
    r.raise_for_status()
    pages = r.json()["query"]["pages"]
    if pages[0].get("missing"):
        return None
    return pages[0]["revisions"][0]["content"]


def extract_roles(wikitext: str) -> list[str]:
    code = mwparserfromhell.parse(wikitext)
    for template in code.filter_templates():
        name = template.name.strip().lower().replace("_", " ")
        if name == "infobox player":
            roles: list[str] = []
            for key in ["roles", "role", "role2"]:
                if template.has(key):
                    value = str(template.get(key).value).strip()
                    if value:
                        roles.extend(
                            r.strip()
                            for r in value.replace("<br>", ",").split(",")
                            if r.strip()
                        )
            return sorted(set(roles))
    return []


def fetch_roles_for_player(
    session: requests.Session, page: str
) -> tuple[list[str], str | None]:
    candidates = [page]
    if page and page[0].islower():
        candidates.append(page[0].upper() + page[1:])

    for candidate in candidates:
        try:
            text = fetch_wikitext(session, candidate)
            if text is not None:
                return extract_roles(text), None
        except requests.HTTPError as e:
            return [], str(e)
        except Exception as e:
            return [], str(e)

    return [], None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    players = load_players(INPUT_CSV)
    print(f"Found {len(players)} unique players in '{INPUT_CSV}'\n")

    session = requests.Session()
    session.headers.update(HEADERS)

    rows: list[dict] = []
    for i, page in enumerate(players, 1):
        roles, error = fetch_roles_for_player(session, page)

        role_str = "|".join(roles) if roles else "unknown"
        row: dict = {"player_page": page, "roles": role_str}
        if error:
            row["error"] = error

        rows.append(row)

        status = f"[{i:>3}/{len(players)}] {page:<20} → {role_str}"
        if error:
            status += f"  ⚠  {error}"
        print(status)

        if i < len(players):
            time.sleep(REQUEST_DELAY)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Done. Results written to '{OUTPUT_CSV}'")


if __name__ == "__main__":
    main()