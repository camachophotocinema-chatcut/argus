# Syndication API field map (worked example)

Endpoint: `https://cdn.syndication.twimg.com/tweet-result?id=<id>&lang=en&token=abc`
Token: any dummy string works (observed `token=abc` returning full JSON). No auth, no cookies. Returns raw JSON — parse with jq.

## Observed fields (tweet 2084564066165825781, 2026-08-04)

- `created_at` — "2026-08-04T08:56:17.000Z"
- `full_text` — full tweet text
- `favorite_count` — 43
- `views.count` — ~6.3K
- `video.duration_millis` — 79110 (79s)
- `video.poster` — https://pbs.twimg.com/amplify_video_thumb/<id>/img/<name>.jpg
- `video.variants` — m3u8 + MP4s: 320x568/632k, 480x852/950k, 720x1280/2176k (9:16 vertical)
- `media.id_str` — 2084563997249142784 (also the yt-dlp video id)

## X-native article id vs tweet id

- Tweet: `x.com/<user>/status/<TWEET_ID>`
- Article card on the tweet links to `x.com/i/article/<ARTICLE_ID>` — a DIFFERENT id, often LOWER than the tweet id (article published before the tweet). Worked example: article 2084538022088122368 vs tweet 2084549593006793187 (the quoting tweet, 2084564066165825781 for the video post).
- Article JSON endpoints (`article-result`, `/article` on cdn.syndication.twimg.com) return empty; the article page is login-gated for anonymous access.

## Reading login-gated X articles — twitter-cli (the working path)

`twitter-cli` (~/.local/bin/twitter, agent-reach, cookie-auth from user's Chrome session) returns **full article text** in the `articleText` field of the tweet that quotes/links the article. This bypasses the login gate entirely — no proxy, no cookies export needed if the user has an active X session in Chrome.

```bash
# 1. Fetch the tweet that links the article (it quotes the article post)
twitter tweet https://x.com/<user>/status/<TWEET_ID> > /tmp/raw.yaml
# 2. The article text is in the quoted tweet's articleText field.
#    Default output is YAML; parse with python yaml (system python3 lacks yaml — use ~/.hermes/hermes-agent/venv/bin/python3)
~/.hermes/hermes-agent/venv/bin/python3 -c "
import yaml
d = yaml.safe_load(open('/tmp/raw.yaml'))
for t in d.get('data', []):
    if t.get('articleText'):
        open('/tmp/article.md','w').write(t['articleText']); break"
```

Pitfalls:
- `twitter article <TWEET_ID>` exists but expects the **tweet id**, not the article id (`x.com/i/article/<ID>` fails with `Article not found`). Use `twitter tweet` instead — it already returns `articleText`.
- `--compact` output is NOT JSON (a JSON parse fails); stick with default YAML and parse via yaml lib.
- The harmless stderr `WARNING twitter_cli.client: Failed to init ClientTransaction` can be ignored — auth still works.
- Verify auth first: `twitter status` → `ok: true` / `authenticated: true`. No `xurl` binary exists on this machine; twitter-cli is the read path.

## yt-dlp on X

- Video URL form that works: `https://x.com/<user>/status/<TWEET_ID>/video/1`
- Captions: `yt-dlp --skip-download --write-subs --sub-langs en --sub-format vtt -o /tmp/caps/video "URL"`
- The syndication JSON's m3u8 needs its `tag=...&v=...` query params to play: `https://video.twimg.com/amplify_video/<id>/pl/<hash>.m3u8?tag=14&v=4c1`

## Reader proxy / search engine behavior (2026-08-04)

- r.jina.ai → `403 AbuseAlleviationError: Anonymous access to domain x.com blocked until <timestamp>` — per-IP cooldown, retry only after the timestamp.
- DuckDuckGo HTML and Bing web search both returned bot-blocked empty pages for an 8h-old article; no mirrors existed yet.
