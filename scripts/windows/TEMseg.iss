; TEMseg.iss — Inno Setup script for TEMseg Windows installer
;
;   Windows zip extraction propagates the Mark-of-the-Web (Zone.Identifier)
;   to every extracted file. .NET Framework refuses to load tagged DLLs,
;   causing Python.Runtime.dll to fail on launch.
;
;   Installers (Inno Setup, NSIS, MSI) do NOT propagate MOTW to extracted
;   files, so this is the recommended distribution method for Windows.
;
; Build prerequisites:
;   - Inno Setup 6.x (https://jrsoftware.org/isdl.php)
;   - TEMseg PyInstaller build already completed (dist\TEMseg\ exists)
;
; Build:
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" TEMseg.iss
;
; Output:
;   dist\TEMseg_Setup.exe
;
; Installation:
;   - Default target: %LOCALAPPDATA%\TEMseg (no admin required)
;   - Creates Start Menu shortcut
;   - Creates optional desktop shortcut

#define MyAppName "TEMseg"
#define MyAppVersion "0.3.1"
#define MyAppPublisher "TEMseg"
#define MyAppExeName "TEMseg.exe"
#define MyAppAssocName MyAppName + " File"
#define MyAppAssocExt ".myp"
#define MyAppAssocKey StringChange(MyAppAssocName, " ", "") + MyAppAssocExt

[Setup]
AppId={{B4A9C1D2-E3F4-4567-8901-2345ABCD6789}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\{#MyAppName}
DisableDirPage=no
DisableProgramGroupPage=yes
OutputDir=..\..\dist
OutputBaseFilename=TEMseg_Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
SetupIconFile=..\..\temseg_icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Recursively include the entire PyInstaller one-dir build
Source: "..\..\dist\TEMseg\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
