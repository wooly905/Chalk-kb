# course-kb (Chalk-kb)

把網頁、文字與字幕等教學素材整理成可追溯出處的課程知識庫，產生 Course Brief，經人工核准後產生逐張投影片規劃（畫面內容、視覺安排、講者講稿與來源）。完整規格見 `Chalk-kb-intro.html`。

## 實作狀態

| 交付階段 | 狀態 |
|---|---|
| 1 · 文字完整流程（text → cards → Brief → 核准 → slides），含版本、證據、核准、快取與續跑 | 已實作，離線測試通過 |
| 2 · 網頁、字幕與影音 | 已實作：`web`、`subtitle`、`media`、`youtube`；YouTube 依「原語言人工字幕 → 原語言自動字幕 → Whisper」選取，空字幕自動改用下一順位；自動字幕與轉錄以 90 秒為組校對（保留原文與時間）。yt-dlp 字幕選取與 faster-whisper 轉錄已以真實工具驗證 |
| 3 · 影格取樣與理解 | 已實作：定時取樣＋場景偵測、本機去重（保留局部新增的板書／公式）、影格理解（Base64 data URL）、與逐字稿依時間整合、卡片與投影片影格引用及圖檔版本驗證；取樣與去重已以真實 ffmpeg 驗證 |
| 驗收（第 15 章） | `accept` 命令可對固定基準做自動檢查、產生逐張人工審核清單並計算工時比例；**實際的人工標註、逐張審核與工時量測尚未進行**，首版驗收需由講師完成後才能宣稱通過 |

尚未以真實模型端點（Azure OpenAI／TPMRouter）驗證內容品質。

上游完整性：`approve` 與 `slides` 會驗證完整依賴鏈（來源 → normalized → chunks → 卡片 → 證據 → 影格圖檔）；任何一層過期都會阻擋，需重跑受影響階段。

## 快速開始

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

$env:OPENAI_API_KEY="..."                         # 只放在環境變數
$env:OPENAI_BASE_URL="https://your-provider.example/v1/"
$env:COURSE_KB_MODEL="your-text-model"            # TPMRouter 時填 auto

copy sources.example.yaml sources.yaml
python -m course_kb run sources.yaml --title "機器學習導論：最佳化" --audience "大學二年級，已修過微積分" --objective "說明梯度下降的更新規則"
# 審核 workspace/brief/brief.md；需要修改時編輯 workspace/brief/brief.json
python -m course_kb brief --render-only
python -m course_kb brief --approve
python -m course_kb slides
```

分階段命令：`ingest sources.yaml [--only ID]`、`chunk`、`cards`、`brief [--force]`、`slides`。全域參數 `--workspace`（預設 `./workspace`）與 `-v`。

結束碼：`0` 完整成功；`1` 有失敗或缺漏（partial），成功成果已保存可續跑；`2` 被阻擋（未核准、核准失效、Brief 過期、設定錯誤）。

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

## 來源欄位補充

- `auto_captions: true`（`subtitle` 類型）：本機字幕檔為自動字幕，`clean: true` 時會校對。
- `keyframes: true` 僅適用 `media`、`youtube`。

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
