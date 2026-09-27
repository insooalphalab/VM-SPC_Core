' 작업 스케줄러가 이 VBScript를 wscript.exe 로 실행한다.
' PowerShell 자체의 "-WindowStyle Hidden" 인자는 Task Scheduler 로 실행할 때 값이
' 누락되는 경우가 실제로 확인됐다(run_daily_pipeline.ps1 상단 주석 참고 + 2026-09-27 실측:
' 등록된 작업 정의에는 -WindowStyle Hidden 이 있었지만 실제 실행된 프로세스의 커맨드라인에는
' 빠져 있었음 — 그 결과 콘솔 창이 그대로 보이다 "응답 없음"으로 표시되고, 클릭하면 빠른 편집
' 모드로 진짜 멈춰버렸다).
'
' WScript.Shell.Run 의 두 번째 인자(0 = SW_HIDE)는 Win32 CreateProcess 의
' STARTUPINFO.wShowWindow 를 직접 지정하는 것이라 훨씬 더 신뢰할 수 있다.
' 세 번째 인자 True 는 완료까지 대기(작업 스케줄러가 "실행 중" 상태를 정확히 추적하도록).

Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
ps1Path = fso.BuildPath(scriptDir, "run_daily_pipeline.ps1")

cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File """ & ps1Path & """"

Set shell = CreateObject("WScript.Shell")
shell.Run cmd, 0, True
