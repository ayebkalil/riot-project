import requests
import json
import time
from pprint import pprint

# ⚠️ REPLACE WITH YOUR NEW API KEY FROM https://developer.riotgames.com/
API_KEY = "RGAPI-bc9bad39-cc7b-41e5-ad0b-5c1b0ac9b0bb"

# Base URLs for different regions
REGIONS = {
    "europe": "https://europe.api.riotgames.com",
    "americas": "https://americas.api.riotgames.com",
    "asia": "https://asia.api.riotgames.com",
    "sea": "https://sea.api.riotgames.com",
    "euw1": "https://euw1.api.riotgames.com",  # Europe West
    "eune1": "https://eune1.api.riotgames.com",  # Europe Nordic & East
    "na1": "https://na1.api.riotgames.com",     # North America
    "kr": "https://kr.api.riotgames.com",       # Korea
    "br1": "https://br1.api.riotgames.com",     # Brazil
    "jp1": "https://jp1.api.riotgames.com",     # Japan
    "la1": "https://la1.api.riotgames.com",     # Latin America North
    "la2": "https://la2.api.riotgames.com",     # Latin America South
    "oc1": "https://oc1.api.riotgames.com",     # Oceania
    "ph2": "https://ph2.api.riotgames.com",     # Philippines
    "sg2": "https://sg2.api.riotgames.com",     # Singapore
    "th2": "https://th2.api.riotgames.com",     # Thailand
    "tr1": "https://tr1.api.riotgames.com",     # Turkey
    "tw2": "https://tw2.api.riotgames.com",     # Taiwan
    "vn2": "https://vn2.api.riotgames.com",     # Vietnam
}

# Region group mappings
REGION_TO_GROUP = {
    "euw1": "europe", "eune1": "europe", "tr1": "europe",
    "na1": "americas", "br1": "americas", "la1": "americas", "la2": "americas",
    "kr": "asia", "jp1": "asia",
    "oc1": "sea", "ph2": "sea", "sg2": "sea", "th2": "sea", "tw2": "sea", "vn2": "sea"
}

def make_request(url, params=None):
    """Make API request with better error handling"""
    if params is None:
        params = {}
    
    # Use header-based authentication (recommended)
    headers = {"X-Riot-Token": API_KEY}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 403:
            print("❌ 403 FORBIDDEN")
            print("Your API key has expired or is invalid!")
            print("\n🔧 Fix:")
            print("  1. Go to https://developer.riotgames.com/")
            print("  2. Login with your Riot account")
            print("  3. Click 'Regenerate API Key'")
            print("  4. Copy the new key and replace API_KEY in this script")
            return None
        elif response.status_code == 404:
            print("❌ 404 NOT FOUND")
            print("Summoner not found in this region. Try a different name or region.")
            return None
        elif response.status_code == 429:
            print("⚠️  429 RATE LIMIT")
            retry_after = int(response.headers.get('Retry-After', 120))
            print(f"Waiting {retry_after} seconds...")
            time.sleep(retry_after)
            return make_request(url, params)
        else:
            print(f"❌ Error {response.status_code}")
            print(f"Message: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None

def test_api_key(region="euw1"):
    """Test if API key works"""
    print("🔍 Testing API key...")
    url = f"{REGIONS[region]}/lol/platform/v3/champion-rotations"
    data = make_request(url)
    if data:
        print("✅ API key is valid!")
        return True
    return False

def get_account_by_riot_id(game_name, tag_line, region_group="europe"):
    """Get account information by Riot ID (gameName#tagLine)"""
    print(f"\n{'='*60}")
    print(f"GETTING ACCOUNT BY RIOT ID: {game_name}#{tag_line}")
    print(f"{'='*60}")

    game_name = requests.utils.quote(game_name)
    tag_line = requests.utils.quote(tag_line)

    url = f"{REGIONS[region_group]}/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
    data = make_request(url)

    if data:
        print("\n📊 ACCOUNT DATA:")
        print(f"  - Game Name: {data.get('gameName')}")
        print(f"  - Tag Line: {data.get('tagLine')}")
        print(f"  - PUUID: {data.get('puuid')[:20]}...")

    return data

def get_ranked_info(summoner_id, region="euw1"):
    """Get ranked information for a summoner"""
    print(f"\n{'='*60}")
    print(f"GETTING RANKED INFO")
    print(f"{'='*60}")
    
    url = f"{REGIONS[region]}/lol/league/v4/entries/by-summoner/{summoner_id}"
    data = make_request(url)
    
    if data:
        if len(data) == 0:
            print("\n🏆 RANKED DATA: Player is unranked")
        else:
            print("\n🏆 RANKED DATA:")
            for queue in data:
                print(f"\n  Queue Type: {queue.get('queueType')}")
                print(f"  - Tier: {queue.get('tier')}")
                print(f"  - Rank: {queue.get('rank')}")
                print(f"  - LP: {queue.get('leaguePoints')}")
                print(f"  - Wins: {queue.get('wins')}")
                print(f"  - Losses: {queue.get('losses')}")
                total_games = queue.get('wins') + queue.get('losses')
                if total_games > 0:
                    win_rate = (queue.get('wins') / total_games * 100)
                    print(f"  - Win Rate: {win_rate:.2f}%")
    
    return data

def get_match_history(puuid, count=5, region="euw1"):
    """Get match history for a player"""
    print(f"\n{'='*60}")
    print(f"GETTING MATCH HISTORY (Last {count} matches)")
    print(f"{'='*60}")
    
    region_group = REGION_TO_GROUP.get(region, "europe")
    url = f"{REGIONS[region_group]}/lol/match/v5/matches/by-puuid/{puuid}/ids"
    params = {"start": 0, "count": count}
    
    match_ids = make_request(url, params)
    
    if match_ids:
        print(f"\n📜 Found {len(match_ids)} matches:")
        for i, match_id in enumerate(match_ids, 1):
            print(f"  {i}. {match_id}")
    
    return match_ids

def get_match_details(match_id, region="euw1"):
    """Get detailed information about a specific match"""
    print(f"\n{'='*60}")
    print(f"GETTING MATCH DETAILS FOR: {match_id}")
    print(f"{'='*60}")
    
    region_group = REGION_TO_GROUP.get(region, "europe")
    url = f"{REGIONS[region_group]}/lol/match/v5/matches/{match_id}"
    data = make_request(url)
    
    if data:
        info = data.get('info', {})
        metadata = data.get('metadata', {})
        
        print("\n🎮 MATCH INFO:")
        print(f"  - Game Mode: {info.get('gameMode')}")
        print(f"  - Game Type: {info.get('gameType')}")
        print(f"  - Duration: {info.get('gameDuration')} seconds ({info.get('gameDuration')//60} minutes)")
        print(f"  - Game Version: {info.get('gameVersion')}")
        print(f"  - Participants: {len(metadata.get('participants', []))}")
        
        # Team info
        teams = info.get('teams', [])
        print("\n👥 TEAMS:")
        for team in teams:
            team_id = team.get('teamId')
            win = "✅ WIN" if team.get('win') else "❌ LOSS"
            print(f"\n  Team {team_id}: {win}")
            print(f"    - Towers: {team.get('objectives', {}).get('tower', {}).get('kills', 0)}")
            print(f"    - Dragons: {team.get('objectives', {}).get('dragon', {}).get('kills', 0)}")
            print(f"    - Barons: {team.get('objectives', {}).get('baron', {}).get('kills', 0)}")
            print(f"    - First Blood: {team.get('objectives', {}).get('champion', {}).get('first', False)}")
            
            # Show bans
            bans = team.get('bans', [])
            if bans:
                banned_champs = [str(ban.get('championId', 'None')) for ban in bans]
                print(f"    - Bans: {', '.join(banned_champs)}")
        
        # Participant details (first 3 players as sample)
        participants = info.get('participants', [])
        print("\n👤 PARTICIPANTS (Sample - First 3 players):")
        for i, participant in enumerate(participants[:3], 1):
            print(f"\n  Player {i}:")
            print(f"    - Summoner: {participant.get('summonerName', 'Unknown')}")
            print(f"    - Champion: {participant.get('championName')}")
            print(f"    - Role: {participant.get('teamPosition')}")
            print(f"    - Level: {participant.get('champLevel')}")
            print(f"    - KDA: {participant.get('kills')}/{participant.get('deaths')}/{participant.get('assists')}")
            kda_ratio = (participant.get('kills') + participant.get('assists')) / max(participant.get('deaths'), 1)
            print(f"    - KDA Ratio: {kda_ratio:.2f}")
            print(f"    - CS: {participant.get('totalMinionsKilled')}")
            print(f"    - Gold: {participant.get('goldEarned'):,}")
            print(f"    - Damage to Champions: {participant.get('totalDamageDealtToChampions'):,}")
            print(f"    - Vision Score: {participant.get('visionScore')}")
            print(f"    - Win: {'✅' if participant.get('win') else '❌'}")
        
        print(f"\n  ... and {len(participants) - 3} more players")
    
    return data

def get_champion_mastery(puuid, region="euw1"):
    """Get champion mastery for a player"""
    print(f"\n{'='*60}")
    print(f"GETTING CHAMPION MASTERY")
    print(f"{'='*60}")
    
    url = f"{REGIONS[region]}/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}/top"
    params = {"count": 5}
    
    data = make_request(url, params)
    
    if data:
        print("\n⭐ TOP 5 CHAMPIONS:")
        for i, champ in enumerate(data, 1):
            print(f"\n  {i}. Champion ID: {champ.get('championId')}")
            print(f"     - Mastery Level: {champ.get('championLevel')}")
            print(f"     - Mastery Points: {champ.get('championPoints'):,}")
    
    return data

def save_data_sample(data, filename):
    """Save data to JSON file for inspection"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Data saved to: {filename}")

def main():
    print("="*60)
    print("RIOT GAMES API DATA EXPLORER - IMPROVED VERSION")
    print("="*60)
    
    # ⚠️ CONFIGURE THESE SETTINGS
    riot_id = "M0ß#EUW"  # Change to any Riot ID (gameName#tagLine)
    region = "euw1"  # kr, euw1, na1, eune1, br1, jp1, etc.
    
    print(f"\n🔍 Searching for Riot ID: {riot_id}")
    print(f"📍 Region: {region.upper()}")
    print(f"\n⚠️  Make sure to:")
    print(f"   1. Replace API_KEY with your key from developer.riotgames.com")
    print(f"   2. Set the correct region for the summoner")
    
    # Test API key first
    if not test_api_key(region):
        print("\n❌ API key test failed. Please fix the issue and try again.")
        return
    
    time.sleep(1.2)
    
    # 1. Get Account Info by Riot ID
    game_name, tag_line = riot_id.split("#")
    account = get_account_by_riot_id(
        game_name,
        tag_line,
        region_group=REGION_TO_GROUP[region]
    )
    
    if not account:
        print("\n❌ Could not find account.")
        print("\n💡 Tips:")
        print("  - Make sure the Riot ID is spelled correctly")
        print("  - Make sure you're using the correct region")
        print("  - Try a different Riot ID")
        print("\n📋 Available regions:")
        for r in sorted(REGIONS.keys()):
            if len(r) <= 5:  # Only show platform regions
                print(f"  - {r}")
        return
    
    time.sleep(1.2)
    
    # 2. Get Match History
    if account:
        match_ids = get_match_history(account['puuid'], count=5, region=region)
        time.sleep(1.2)
    
    # 4. Get Detailed Match Info (first match only)
    if match_ids and len(match_ids) > 0:
        match_details = get_match_details(match_ids[0], region=region)
        
        # Save full match data to file for detailed inspection
        if match_details:
            save_data_sample(match_details, "sample_match_data.json")
        time.sleep(1.2)
    
    # 4. Get Champion Mastery
    if account:
        mastery_data = get_champion_mastery(account['puuid'], region)
    
    print("\n" + "="*60)
    print("✅ EXPLORATION COMPLETE!")
    print("="*60)
    print("\n💡 Next Steps:")
    print("  1. Check 'sample_match_data.json' for full match data structure")
    print("  2. Modify summoner_name and region variables to explore different players")
    print("  3. Use this data to plan your ML features")
    print("\n📋 Key Data Points Available:")
    print("  ✓ Player stats (KDA, CS, damage, gold)")
    print("  ✓ Champion information")
    print("  ✓ Team compositions")
    print("  ✓ Match outcomes")
    print("  ✓ Objectives (towers, dragons, barons)")
    print("  ✓ Bans")
    print("  ✓ Timeline data (can be fetched separately)")
    print("  ✓ Ranked tier information")
    print("  ✓ Champion mastery")
    print("\n🎯 For your ML project:")
    print("  → Collect 5,000-10,000 matches")
    print("  → Extract ~100+ features per match")
    print("  → Target: Predict win/loss")

if __name__ == "__main__":
    main()