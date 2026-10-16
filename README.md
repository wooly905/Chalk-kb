# course-kb (Chalk-kb)

把網頁、文字、字幕與影音等教學素材整理成可追溯出處的課程知識庫，產生 Course Brief，核准後產生逐張投影片規劃（畫面內容、視覺安排、講者講稿與來源），並匯出 PowerPoint（`.pptx`）。完整規格見 `Chalk-kb-intro.html`。

## 實作狀態

| 交付階段 | 狀態 |
|---|---|
| 1 · 文字完整流程（text → cards → Brief → 核准 → slides），含版本、證據、核准、快取與續跑 | 已實作，離線測試通過 |
| 2 · 網頁、字幕與影音 | 已實作：`web`、`subtitle`、`media`、`youtube`；YouTube 依「原語言人工字幕 → 原語言自動字幕 → Whisper」選取，空字幕自動改用下一順位；自動字幕與轉錄以 90 秒為組校對（保留原文與時間）。yt-dlp 字幕選取與 faster-whisper 轉錄已以真實工具驗證 |
| 3 · 影格取樣與理解 | 已實作：定時取樣＋場景偵測、本機去重（保留局部新增的板書／公式）、影格理解（Base64 data URL）、與逐字稿依時間整合、卡片與投影片影格引用及圖檔版本驗證；取樣與去重已以真實 ffmpeg 驗證 |
| PPTX 匯出與一鍵流程 | 已實作：`build` 一個指令跑完全部階段並輸出 `slides.pptx`；`export-pptx` 可單獨匯出；partial 時仍輸出並在品質報告頁標示缺漏 |
| 驗收（第 15 章） | `accept` 命令可對固定基準做自動檢查、產生逐張人工審核清單並計算工時比例；**實際的人工標註、逐張審核與工時量測尚未進行**，首版驗收需由講師完成後才能宣稱通過 |

尚未以真實模型端點（Azure OpenAI／TPMRouter）驗證內容品質。

上游完整性：`approve` 與 `slides` 會驗證完整依賴鏈（來源 → normalized → chunks → 卡片 → 證據 → 影格圖檔）；任何一層過期都會阻擋，需重跑受影響階段。

## 快速開始

### 1. 安裝

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

處理 `media` 或 `youtube` 來源時，另需安裝 `ffmpeg` 並放在 PATH。

### 2. 設定語言模型（只放在環境變數，每開一個新終端機都要重設）

```powershell
$env:OPENAI_API_KEY="..."
$env:OPENAI_BASE_URL="https://your-provider.example/v1/"
$env:COURSE_KB_MODEL="your-text-model"            # TPMRouter 時填 auto
```

### 3. 準備 sources.yaml

```powershell
copy sources.example.yaml sources.yaml
```

```yaml
course:                          # 選填；命令列 --title/--audience/--objective 會覆寫
  title: always on agents
  audience: software engineers
  objectives:
    - learn how always-on agents work
sources:
  - id: notes-01                 # 必填，唯一；只能用英數與 - _ .
    type: text                   # 必填：web | text | subtitle | media | youtube
    location: ./materials/notes01.md   # 必填：本機路徑（以 sources.yaml 所在目錄為準）或網址
    license: own                 # 選填，預設 unknown
  - id: yt-01
    type: youtube
    location: https://www.youtube.com/watch?v=XXXX
    rights_confirmed: true       # youtube 必填 true，表示你有權使用該影片
    keyframes: true              # 選填，僅 media / youtube
```

其他選填欄位：`title`、`language`、`clean`（預設 true，字幕校對）、`auto_captions`（`subtitle` 類型的本機字幕為自動字幕時設 true）。

範例檔中的來源路徑與網址皆不存在，必須換成真實素材，否則 ingest 失敗、卡片為 0。

### 4. 一個指令產生投影片

```powershell
python -m course_kb build sources.yaml
```

依序執行 ingest → chunk → cards → brief → 自動核准 → slides → 匯出 `slides.pptx` 到**目前執行目錄**（可用 `--out PATH` 指定其他路徑）。

- 自動核准會替未解決的先備寫入 `audience_has`、替循環邊寫入 `remove_edge`、替失敗項目寫入 `accept_missing`，理由皆標示「build 自動決定，未經人工審核」，記錄在 `workspace/brief/brief.json`。
- partial 仍會輸出 pptx：第二頁起為「品質報告」，列出生成失敗、未涵蓋卡片、上游缺漏、自動決定與待確認項目；失敗的投影片以紅色佔位頁保留位置，有待確認項目的投影片標題加 `△`。
- 講者講稿、學習目標、來源與待確認項目寫在每張投影片的備忘稿。
- 已完成的階段會快取，重跑 `build` 只補做缺漏或變更的部分；`--force` 忽略快取重跑。

## 人工審核流程

需要逐項把關時，用 `--stop-before-approve` 停在 Course Brief：

```powershell
python -m course_kb build sources.yaml --stop-before-approve   # 或 run sources.yaml
# 審核 workspace/brief/brief.md；需要修改時編輯 workspace/brief/brief.json 的 review_decisions
python -m course_kb brief --render-only
python -m course_kb brief --approve
python -m course_kb slides
python -m course_kb export-pptx
```

分階段命令：`ingest sources.yaml [--only ID]`、`chunk`、`cards`、`brief [--force]`、`slides`、`export-pptx [--out PATH]`。全域參數 `--workspace`（預設 `./workspace`）與 `-v`。所有命令都要加 `python -m course_kb` 前綴。

結束碼：`0` 完整成功；`1` 有失敗或缺漏（partial），成功成果已保存可續跑；`2` 被阻擋（未核准、核准失效、Brief 過期、設定錯誤、缺少 API 金鑰）。

## 輸出位置

| 檔案 | 內容 |
|---|---|
| `./slides.pptx`（執行目錄） | 投影片與品質報告 |
| `workspace/brief/brief.md`、`brief.json` | Course Brief（閱讀版／可編輯主檔） |
| `workspace/slides/slides.md`、`slides.json` | 逐張投影片規劃（閱讀版／結構化） |
| `workspace/reports/run.json` | 各階段完成狀態與失敗原因 |

## 審核決定（brief.json → `review_decisions`）

每筆需要 `item_id`、`decision` 與非空的 `reason`；`item_id` 取自 `brief.md`。

| decision | 用於 | 說明 |
|---|---|---|
| `map_prerequisite` | `prereq:…` | 加上 `target_card_id`，把未對應的先備指定到現有卡片 |
| `audience_has` | `prereq:…` | 受眾已具備此先備知識 |
| `need_material` | `prereq:…` | 需要補充素材；在補齊並重跑前不能核准 |
| `remove_edge` | `edge:A->B` | 人工移除先備邊（例如解開循環） |
| `accept_missing` | `ingest:…`、`cards:…` 等失敗項目 | 接受缺漏；輸出仍標示 partial |
| `acknowledge` | 任何項目 | 記錄已閱讀與處理說明 |

投影片排除：`excluded_slide_cards: [{"card_id": "k003", "reason": "受眾已修過微積分"}]`。

`map_prerequisite` 不可指向該先備所屬的卡片本身，且必須形成有效的先備邊（未被移除、未造成循環）才能核准。

## 驗收

```powershell
python -m course_kb accept acceptance\baseline.example.yaml --review acceptance\review.example.yaml --timelog acceptance\timelog.example.csv
```

輸出 `workspace/reports/acceptance.json`、`acceptance.md` 與 `review.template.yaml`。自動檢查、人工逐張審核與工時（工具輔助 ≤ 純人工 50%）三者皆通過才回傳 0。

- 驗收先確認目前核准仍有效、`slides.json` 綁定該核准且與逐張驗證結果一致；Brief、卡片或投影片變更後不會沿用舊結果。
- 每筆人工審核需填 `content_hash`（由 `review.template.yaml` 提供），綁定實際受審的投影片內容與核准版本；內容變更後必須重新審核。

影音快取：外層快取同時驗證引用的影格圖檔；Whisper、校對與影格理解的內層快取使用與外層相同的設定識別（含 API 根網址、Schema、提示詞與處理器版本）。每個成功的校對時間窗與影格描述會立即保存，失敗後重跑只補做缺漏。`ingest --force` 會重新轉錄、校對與分析影格，並記錄實際模型。

## 測試

```powershell
python -m pytest -q
```

測試使用假的 LLM 與 embedding，不需要網路或 API 金鑰。
