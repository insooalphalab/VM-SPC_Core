# 매일 장마감 후(16:30 KST) 실행되는 EOD 배치 — Windows 작업 스케줄러("VM-SPC Core 일일 업데이트")가 이 스크립트를 호출한다.
#
# vm_spc/pipeline.py 만 단독으로 돌리면 안 된다 — 그 파일은 이미 저장된 data/*.csv 를 읽기만 하고
# API 를 호출하지 않는다(파일 docstring 참고). 새 일봉을 실제로 받아오려면 Stage1 수집이 포함된
# vm_predict/v2_run.py 를 먼저 돌려야 한다. 순서:
#   1) vm_predict/v2_run.py   — Stage1(일봉 수집) → Stage2(Legacy/CUSUM/T²) → Stage3(VM 탭 렌더링)
#   2) pair_spc/run_pair_spc.py — 대표종목 실증 + 공적분 SPC → PAIR-SPC 탭 렌더링
#   2.5) dart_events/collect.py --recent — DART 최근 2년 재무·공시 + 바스켓 PER/PBR 상태(인덱스 카드용)
#   2.7) reports/run.py — 텔레그램 리포트 요약 새 글 → 섹터 리포트 흐름(배경 자료, 인덱스 맨 아래)
#   3) vm_spc/pipeline.py     — 챔피언-챌린저 Walk-Forward 재검증 → 탭 렌더링 → results/index.html 자동 갱신
#   3.5) scenario/render_risk.py — 박스권 손절·수량 페이지(scenario_targets.json 종목, 수급 증분 갱신)
#   3.6) scenario/screen.py — 오늘의 후보: 코스피 시총 상위 200에서 장세별 전략표에 맞는 규칙 충족 종목(목록별 상위 5)
#   4) results/ → 구글 드라이브 동기화 폴더로 미러링(robocopy /MIR) — 폰에서 구글 드라이브 앱으로
#      상세 대시보드까지 열람 가능하게. index.html 의 링크는 상대경로라 results/ 폴더 구조가 통째로
#      옆에 있어야 클릭이 된다(단, 드라이브 모바일 앱이 그 상대링크 이동을 보장하진 않는다 — 그래도
#      백업/개별 파일 열람 용도로는 충분하다는 걸 확인하고 2026-09-27에 추가함).
#   5) vm_spc/notify_telegram.py — ① ETF 요약 + index.html, ② 관심 종목 전략 요약 + risk_scenarios.html, ③ 오늘의 후보 + screen.html 을 텔레그램으로 전송
#
# 실행 로그는 logs/pipeline_YYYYMMDD_HHMMSS.log 에 UTF-8로 남는다(.gitignore 의 *.log 에 이미 포함).
#
# 창/인코딩 이력(2026-09-27, 실사용 중 재현된 문제들 — 같은 실수를 반복하지 않기 위한 기록):
#  1) 작업 스케줄러의 Settings.Hidden 은 "작업이 스케줄러 목록에서 안 보인다"는 뜻이지 "실행되는
#     프로세스 창이 안 보인다"는 뜻이 아니다. 프로세스 창 자체를 없애려면 액션을 실행하는 지점에서
#     WindowStyle 을 제어해야 한다.
#  2) cmd.exe 로 파이썬을 감싸 리다이렉트하면(`cmd /c "python ... >> log 2>&1"`) cmd.exe 가 자기
#     콘솔 창을 새로 띄운다 — 부모 PowerShell 을 숨겨도 이 자식 창은 그대로 보인다. 그 창 안을
#     클릭하면 Windows 콘솔의 "빠른 편집 모드"가 프로세스 전체를 일시정지시킨다("멈춘 것처럼"
#     보이는 증상의 원인). 그래서 cmd.exe 를 아예 안 거친다 — Start-Process 로 python.exe 를
#     직접, WindowStyle Hidden 으로 실행하고 표준출력/에러를 파일로 직접 리다이렉트한다. 이건
#     OS 레벨 파일 리다이렉트라 PowerShell 의 콘솔 인코딩 왕복 변환(한글 깨짐)이나
#     NativeCommandError 래핑(이전 버전에서 겪은 문제) 문제도 같이 사라진다.
#  3) 작업 스케줄러 액션에 `powershell.exe ... -WindowStyle Hidden -File ...` 을 그대로 등록해도,
#     실제 실행 시점에 Task Scheduler 가 -WindowStyle Hidden 인자를 누락시키는 게 실측으로
#     확인됐다(등록된 작업 정의에는 있었지만 실행된 프로세스의 실제 커맨드라인에는 빠져 있었음).
#     그 결과 이 스크립트를 감싸는 바깥쪽 PowerShell 콘솔 창 자체가 숨겨지지 않고 그대로 남아
#     "응답 없음"으로 보이다 클릭 시 진짜로 멈추는 문제가 재발했다. 그래서 작업 스케줄러의
#     액션은 이제 이 .ps1 을 직접 호출하지 않고, `run_daily_pipeline_hidden.vbs` 를 통해
#     `WScript.Shell.Run(cmd, 0, True)` 로 띄운다 — 이건 Win32 STARTUPINFO.wShowWindow 를 직접
#     지정하는 방식이라 PowerShell 자체의 -WindowStyle 인자보다 훨씬 신뢰할 수 있다.
#  4) 이 파일은 반드시 UTF-8 "BOM 포함"으로 저장해야 한다. BOM 이 없으면 Windows PowerShell 5.1 이
#     시스템 코드페이지(CP949)로 읽어서 한글 경로("G:\내 드라이브")가 깨지고 드라이브 동기화가 실패한다.

$root = "C:\Users\USER\Desktop\New toy\VM-SPC_Core"
Set-Location $root

$logDir = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $logDir "pipeline_$stamp.log"
$pythonExe = "C:\Users\USER\AppData\Local\Programs\Python\Python312\python.exe"

function Write-Utf8Line($text) {
    [System.IO.File]::AppendAllText($logFile, "$text`r`n", [System.Text.Encoding]::UTF8)
}

function Run-Step($label, $relScriptPath, $extraArgs = @()) {
    Write-Utf8Line "`n===== [$label] START $(Get-Date -Format o) ====="
    $scriptFull = Join-Path $root $relScriptPath
    $tmpOut = Join-Path $logDir "_tmp_$stamp`_out.log"
    $tmpErr = Join-Path $logDir "_tmp_$stamp`_err.log"

    # Start-Process -WindowStyle Hidden : 창을 아예 안 띄운다(cmd.exe 를 거치지 않으므로 자식
    # 창도 없다). -RedirectStandardOutput/-Error 는 OS 파일 리다이렉트라 인코딩 왕복이 없다.
    #
    # -ArgumentList 에 배열을 그대로 넘기면 요소별 자동 인용이 안 돼서, 경로에 공백이 있으면
    # (이 프로젝트 경로 "New toy" 처럼) 거기서 인자가 잘린다 — 그래서 하나의 문자열로 직접
    # 조립하면서 스크립트 경로만 큰따옴표로 감싼다(옵션 인자는 공백이 없어 그대로 이어붙임).
    $argStr = '"' + $scriptFull + '"'
    foreach ($a in $extraArgs) { $argStr += " $a" }
    $proc = Start-Process -FilePath $pythonExe -ArgumentList $argStr `
        -WorkingDirectory $root -WindowStyle Hidden -Wait -PassThru `
        -RedirectStandardOutput $tmpOut -RedirectStandardError $tmpErr
    $rc = $proc.ExitCode

    if (Test-Path $tmpOut) {
        Get-Content $tmpOut -Raw -Encoding UTF8 | ForEach-Object { [System.IO.File]::AppendAllText($logFile, $_, [System.Text.Encoding]::UTF8) }
        Remove-Item $tmpOut -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path $tmpErr) {
        Get-Content $tmpErr -Raw -Encoding UTF8 | ForEach-Object { [System.IO.File]::AppendAllText($logFile, $_, [System.Text.Encoding]::UTF8) }
        Remove-Item $tmpErr -Force -ErrorAction SilentlyContinue
    }

    Write-Utf8Line "===== [$label] END (exit=$rc) $(Get-Date -Format o) ====="
    return $rc
}

$env:PYTHONIOENCODING = "utf-8"

$rc1 = Run-Step "STAGE1-3 vm_predict (forecast model)"    "vm_predict\v2_run.py"
$rc2 = Run-Step "PAIR-SPC (representative stock corr.)"   "pair_spc\run_pair_spc.py"
# DART 재무·공시(최근 2년) + 바스켓 PER/PBR 상태 — 인덱스 카드가 읽으므로 vm_spc(인덱스 생성)보다 먼저.
$rcD = Run-Step "DART financials + valuation"             "dart_events\collect.py" @("--recent")
# 리포트 요약(텔레그램 공개 채널) 새 글 → 섹터 리포트 흐름(배경 자료) — 인덱스 맨 아래 접힌 한 줄이 이 요약을 읽으므로 인덱스보다 먼저.
$rcR = Run-Step "analyst reports (reference)"             "reports\run.py"
$rc3 = Run-Step "vm_spc champion-challenger + index"       "vm_spc\pipeline.py"
# 박스권 손절·수량 페이지(scenario_targets.json 종목, 수급 증분 갱신 포함) — 드라이브 동기화 전에 만들어 폰에서도 최신본
$rcS = Run-Step "scenario risk page"                      "scenario\render_risk.py"
# 오늘의 후보 200종목 프로그램매매 증분(승률 모형의 이탈일 프로그램 z, 9.79) — 실패해도 후보 페이지는 평균값으로 계산
$rcG = Run-Step "program trading (screen set, daily)"     "stock_track\collect_program.py" @("--screen")
# 오늘의 후보: 코스피 시총 상위 200에서 현재 장세 전략표(scenario/screen.py PLAYBOOK)에 맞는 규칙 충족 종목
$rcC = Run-Step "screen candidates (KOSPI 200)"          "scenario\screen.py"

Write-Utf8Line "`nALL DONE: vm_predict=$rc1 pair_spc=$rc2 dart=$rcD reports=$rcR vm_spc=$rc3 scenario=$rcS screen=$rcC"

# 구글 드라이브 동기화 폴더(G:\내 드라이브, Drive for Desktop)로 results/ 를 통째로 미러링.
# robocopy /MIR 는 소스에 없는 파일은 대상에서도 지워서 완전히 최신 상태로 맞춘다. G: 드라이브가
# 없으면(드라이브 앱 로그아웃/제거 등) 조용히 건너뛴다 — 이건 핵심 파이프라인이 아니라 보조 기능.
$driveRoot = "G:\내 드라이브\VM-SPC_Core"
if (Test-Path "G:\") {
    Write-Utf8Line "`n===== [Google Drive 동기화] START $(Get-Date -Format o) ====="
    New-Item -ItemType Directory -Force -Path $driveRoot | Out-Null
    robocopy (Join-Path $root "results") (Join-Path $driveRoot "results") /MIR /NFL /NDL /R:1 /W:2 | Out-Null
    $rcSync = $LASTEXITCODE   # robocopy: 0-7 = 성공(변경 종류별 코드), 8 이상 = 실패
    # basket_watchlist.json 은 .gitignore 대상이라(깃허브에 올리지 않음) 드라이브가 유일한 백업이다.
    Copy-Item (Join-Path $root "basket_watchlist.json") (Join-Path $driveRoot "basket_watchlist.json") -Force -ErrorAction SilentlyContinue
    # scenario_targets.json(박스권 페이지 대상 종목)도 같은 이유로 드라이브에만 백업
    Copy-Item (Join-Path $root "scenario_targets.json") (Join-Path $driveRoot "scenario_targets.json") -Force -ErrorAction SilentlyContinue
    Write-Utf8Line "===== [Google Drive 동기화] END (robocopy exit=$rcSync) $(Get-Date -Format o) ====="
} else {
    Write-Utf8Line "`n===== [Google Drive 동기화] SKIP — G: 드라이브 연결 안 됨 $(Get-Date -Format o) ====="
}

# 앞 3단계 중 하나라도 실패했으면 텔레그램 메시지 맨 위에 경고를 붙인다(그래도 전송은 한다 —
# 뭐가 됐는지 안 된 건지 아는 게 아무 소식 없는 것보다 낫다).
$runStatus = if (($rc1 -eq 0) -and ($rc2 -eq 0) -and ($rcD -eq 0) -and ($rcR -eq 0) -and ($rc3 -eq 0) -and ($rcS -eq 0) -and ($rcC -eq 0)) { "ok" } else { "warn" }
$rc4 = Run-Step "notify_telegram" "vm_spc\notify_telegram.py" @("--run-status", $runStatus)

# 금요일만: 검증용 약 790종목 프로그램매매·신용잔고 증분(한 번 호출에 30일치라 주 1회로 충분, 약 30분) — 알림 뒤에 돌려 지연 없게.
# 관심 종목·오늘의 후보는 위 페이지 단계가 매일 갱신한다.
if ((Get-Date).DayOfWeek -eq [DayOfWeek]::Friday) {
    $rcP = Run-Step "program trading (validation set, weekly)" "stock_track\collect_program.py" @("--validation")
    $rcK = Run-Step "credit balance (validation set, weekly)"  "stock_track\collect_credit.py" @("--validation")
    # 검증용 종목 분기 재무(오늘의 후보 돌파 목록의 매출·이익 표시, 9.78) — 약 3분
    $rcF = Run-Step "DART financials (validation set, weekly)" "dart_events\collect.py" @("--validation")
}

# 로그 30일 이상 지난 건 정리
Get-ChildItem -Path $logDir -Filter "pipeline_*.log" |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) } |
    Remove-Item -Force -ErrorAction SilentlyContinue
