import json
import random
import re
import time
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from fake_useragent import UserAgent
except Exception:
    UserAgent = None

# ================= CONFIG =================

REGION = "euw"
BASE_URLS = [
    "https://www.op.gg/leaderboards/tier",
    "https://op.gg/leaderboards/tier",
    "https://www.op.gg/lol/leaderboards/tier",
    "https://op.gg/lol/leaderboards/tier",
]
OUTPUT_DIR = Path("data/opgg")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
DEBUG_DIR = OUTPUT_DIR / "debug"
DEBUG_DIR.mkdir(exist_ok=True, parents=True)

TARGETS = {
    "iron": 1000,
    "bronze": 1000,
    "silver": 1000,
    "gold": 1000,
    "platinum": 1000,
    "emerald": 1000,
    "diamond": 1000,
    "master": 1000,
    "grandmaster": 700,
    "challenger": 300
}

FALLBACK_UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

ua = UserAgent() if UserAgent else None
DELAY_MIN, DELAY_MAX = 1.5, 3.5

# ================= SCRAPER =================

class OpggTierScraper:
    def __init__(self, region="euw"):
        self.region = region
        self.session = requests.Session()
        self.collected = []

        retries = Retry(
            total=3,
            backoff_factor=0.6,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def get_headers(self):
        return {
            "User-Agent": ua.random if ua else random.choice(FALLBACK_UAS),
            "Accept": "text/html",
            "Accept-Language": "en-US,en;q=0.9",
        }

    def scrape_tier(self, tier, target):
        print(f"\n[*] Scraping {tier.upper()} -> target: {target}")
        page = 1
        total = 0

        while total < target:
            print(f"  [page {page}] ...", end="")
            params = {
                "tier": tier,
                "page": page,
                "region": self.region,
            }

            response = None
            for base_url in BASE_URLS:
                try:
                    response = self.session.get(base_url, params=params, headers=self.get_headers(), timeout=15)
                except requests.RequestException as exc:
                    print(f" failed ({exc})")
                    response = None
                    continue

                if response.status_code == 200:
                    break

            if not response or response.status_code != 200:
                print(f" failed (status={response.status_code})")
                break

            players = self.parse_page(response.text, tier)
            if not players:
                self.dump_debug_html(response.text, tier, page, response.url)
                print(" no more players found")
                break

            for p in players:
                if total >= target:
                    break
                self.collected.append(p)
                total += 1

            print(f" collected {total}/{target}")

            page += 1
            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

        print(f"[OK] Done {tier.upper()} -> {total}")
        return

    def parse_page(self, html, tier=None):
        soup = BeautifulSoup(html, "html.parser")
        players = []

        # ==== UPDATE SELECTORS BASED ON CURRENT OP.GG HTML ====

        if self.looks_like_blocked(html):
            print("[!] Possible bot-block/CAPTCHA page returned")
            return []

        rows = soup.select("table.ranking-table tbody tr")
        if not rows:
            players = self.parse_from_next_data(soup)
            if players:
                return players
            players = self.parse_from_streamed_html(html, tier)
            if players:
                return players
            print("[!] Could not find rows — check selectors or page content!")
            return []

        for row in rows:
            try:
                name_tag = (
                    row.select_one("a[href*='/summoner']")
                    or row.select_one("a[href*='/summoners']")
                    or row.select_one("td a")
                )
                if not name_tag:
                    continue
                summoner = name_tag.text.strip()

                rank_elem = row.select_one("td:nth-child(2)")
                rank = rank_elem.text.strip() if rank_elem else ""

                lp_elem = row.select_one("td:nth-child(3)")
                lp = lp_elem.text.strip() if lp_elem else ""

                winrate_elem = row.select_one("td:nth-child(4)")
                winrate = winrate_elem.text.strip() if winrate_elem else ""

                players.append({
                    "summoner_name": summoner,
                    "rank_tier": rank,
                    "lp": lp,
                    "win_rate": winrate,
                    "region": self.region
                })
            except Exception as e:
                print(f"[ERROR] Parse error: {e}")
                continue

        return players

    def parse_from_next_data(self, soup):
        script = soup.select_one("script#__NEXT_DATA__")
        if script and script.string:
            try:
                data = json.loads(script.string)
                return self.extract_players_from_json(data)
            except Exception:
                return []

        for script in soup.find_all("script"):
            text = script.string or script.text or ""
            if "summonerName" not in text:
                continue
            json_text = self.extract_json_object(text)
            if not json_text:
                continue
            try:
                data = json.loads(json_text)
                players = self.extract_players_from_json(data)
                if players:
                    return players
            except Exception:
                continue

        return []

    def extract_json_object(self, text):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        return match.group(0)

    def extract_players_from_json(self, data):
        players = []
        seen = set()

        def normalize(value):
            if value is None:
                return ""
            if isinstance(value, (int, float)):
                return str(value)
            return str(value).strip()

        def walk(obj):
            if isinstance(obj, dict):
                if "summonerName" in obj or "summoner" in obj:
                    summoner = obj.get("summonerName") or obj.get("summoner")
                    if summoner:
                        summoner = normalize(summoner)
                        if summoner and summoner not in seen:
                            seen.add(summoner)
                            players.append({
                                "summoner_name": summoner,
                                "rank_tier": normalize(obj.get("tier") or obj.get("rankTier") or obj.get("division") or obj.get("rank")),
                                "lp": normalize(obj.get("lp") or obj.get("leaguePoints") or obj.get("points")),
                                "win_rate": normalize(obj.get("winRate") or obj.get("winrate") or obj.get("win_rate")),
                                "region": self.region,
                            })
                for value in obj.values():
                    walk(value)
            elif isinstance(obj, list):
                for item in obj:
                    walk(item)

        walk(data)
        return players

    def looks_like_blocked(self, html):
        text = html.lower()
        tokens = ["captcha", "cf-chl", "cloudflare", "attention required", "access denied"]
        return any(token in text for token in tokens) and len(text) < 30000

    def parse_from_streamed_html(self, html, tier=None):
        players = []
        seen = set()

        # Extract summoner data with rank info from Next.js streaming chunks
        summoner_pattern = rf"/lol/summoners/{re.escape(self.region)}/([^\"\s\"]+)"
        matches = re.finditer(summoner_pattern, html)
        
        for i, match in enumerate(matches):
            decoded = unquote(match.group(1))
            summoner = self.format_summoner_name(decoded)
            if summoner and summoner not in seen:
                seen.add(summoner)
                
                # Get text around the summoner link to extract rank/LP/WR
                start = max(0, match.start() - 500)
                end = min(len(html), match.end() + 500)
                context = html[start:end]
                
                rank_tier = tier if tier else self.extract_rank_from_context(context)
                lp = self.extract_lp_from_context(context)
                win_rate = self.extract_winrate_from_context(context)
                
                players.append({
                    "summoner_name": summoner,
                    "rank_tier": rank_tier,
                    "lp": lp,
                    "win_rate": win_rate,
                    "region": self.region,
                })

        return players
    
    def extract_rank_from_context(self, context):
        # Look for tier names: Iron, Bronze, Silver, Gold, Platinum, Emerald, Diamond, Master, GrandMaster, Challenger
        tiers = ["Challenger", "GrandMaster", "Master", "Diamond", "Emerald", "Platinum", "Gold", "Silver", "Bronze", "Iron"]
        for tier in tiers:
            if tier in context:
                return tier
        return ""
    
    def extract_lp_from_context(self, context):
        # Look for LP patterns like "123" after "LP" or numerical values
        lp_match = re.search(r'(\d+)\s*(?:LP|lp|leaguePoints)', context)
        if lp_match:
            return lp_match.group(1)
        # Try to find standalone numbers that look like LP values (2-4 digits)
        numbers = re.findall(r'\b(\d{1,3})\b', context)
        if numbers:
            return numbers[-1]  # Get the last number found
        return ""
    
    def extract_winrate_from_context(self, context):
        # Look for win rate patterns like "52.5%" or "52"
        wr_match = re.search(r'(\d+(?:\.\d+)?)\s*%', context)
        if wr_match:
            return wr_match.group(1) + "%"
        return ""

    def format_summoner_name(self, slug):
        if "-" in slug:
            name, tag = slug.rsplit("-", 1)
            if name and tag:
                return f"{name}#{tag}"
        return slug


    def dump_debug_html(self, html, tier, page, url):
        safe_tier = re.sub(r"[^a-z0-9_-]", "_", tier.lower())
        file_path = DEBUG_DIR / f"{safe_tier}_page_{page}.html"
        meta_path = DEBUG_DIR / f"{safe_tier}_page_{page}.txt"
        file_path.write_text(html, encoding="utf-8")
        meta_path.write_text(f"URL: {url}\n", encoding="utf-8")

    def save(self):
        df = pd.DataFrame(self.collected).drop_duplicates("summoner_name")
        df.to_csv(OUTPUT_DIR / "opgg_leaderboards.csv", index=False)
        print(f"\n[SAVE] Saved {len(df)} players to CSV")

# ================= MAIN =================

def main():
    scraper = OpggTierScraper(REGION)

    for tier, count in TARGETS.items():
        scraper.scrape_tier(tier, count)

    scraper.save()

if __name__ == "__main__":
    main()
