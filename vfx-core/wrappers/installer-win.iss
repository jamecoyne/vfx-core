; The Windows installer for one effect, named after it, with the workspace's other effects as
; options. The user picks the apps: OpenFX (Resolve, Nuke, Natron), Resolume Arena and Avenue
; (FFGL) and After Effects / Premiere Pro, preselected when Setup finds them.
; package-win.sh builds the plugins, writes Effects (an .iss that includes
; installer-effect-win.iss once per effect) and runs this with the lead effect's names:
;   ISCC -DVersion=1.2.3 -DSource=<built plugins> -DEffects=<effects.iss> -DAppName=...
;        -DLeadFile=... -DPublisher=... -DCategory=... -DAppGuid=... installer-win.iss
; -DTestRoot=<folder> installs everything under that folder instead, without administrator
; rights, for testing the installer.

#ifndef AppName
  #error Build this with package-win.sh, which passes the effects and their names.
#endif
#ifndef Version
  #define Version "dev"
#endif

#ifdef TestRoot
  #define OfxPlugins TestRoot + "\OFX\Plugins"
  #define MediaCore TestRoot + "\MediaCore"
  #define Documents TestRoot + "\Documents"
  #define AppDir TestRoot + "\App"
#else
  #define OfxPlugins "{commoncf64}\OFX\Plugins"
  #define MediaCore "{commonpf64}\Adobe\Common\Plug-ins\7.0\MediaCore"
  #define Documents "{userdocs}"
  #define AppDir AddBackslash("{commonpf64}") + AppName
#endif
#define Arena Documents + "\Resolume Arena\Extra Effects"
#define Avenue Documents + "\Resolume Avenue\Extra Effects"

[Setup]
AppId={{{#AppGuid}}
AppName={#AppName}
AppVersion={#Version}
AppPublisher={#Publisher}
UninstallDisplayName={#AppName} plugins
; The folder only holds the uninstaller; the plugins go where each app looks.
DefaultDirName={#AppDir}
DisableDirPage=yes
DisableProgramGroupPage=yes
#ifdef TestRoot
PrivilegesRequired=lowest
#else
PrivilegesRequired=admin
#endif
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Light or dark, as Windows is set.
WizardStyle=modern dynamic
Compression=lzma2
SolidCompression=yes
; Hosts load .ofx and .aex files too, so those count as files in use.
CloseApplicationsFilter=*.exe,*.dll,*.chm,*.ofx,*.aex

[Messages]
SelectComponentsDesc=Which apps should {#AppName} be added to?
SelectComponentsLabel2=Installed apps are already checked.

; One type, so that the page shows only the list and not Inno's own Full and Compact.
[Types]
Name: "custom"; Description: "Custom"; Flags: iscustom

; A plugin is installed when its app is chosen, and an example when the example is too.
[Components]
Name: "ofx"; Description: "DaVinci Resolve, Nuke and Natron (OpenFX)"
Name: "arena"; Description: "Resolume Arena"
Name: "avenue"; Description: "Resolume Avenue"
Name: "adobe"; Description: "After Effects and Premiere Pro"

; v0.1.0 installed segcam as SegcamMotionRust. Its files go for each app chosen, so no app
; keeps the old one. The binaries are named on their own so that Setup's check for files in
; use sees them, and closes an app that has one loaded.
[InstallDelete]
Type: files; Name: "{#OfxPlugins}\SegcamMotionRust.ofx.bundle\Contents\Win64\SegcamMotionRust.ofx"; Components: ofx
Type: filesandordirs; Name: "{#OfxPlugins}\SegcamMotionRust.ofx.bundle"; Components: ofx
Type: files; Name: "{#MediaCore}\SegcamMotionRust\SegcamMotionRust.aex"; Components: adobe
Type: filesandordirs; Name: "{#MediaCore}\SegcamMotionRust"; Components: adobe
Type: files; Name: "{#Arena}\SegcamMotionRust.dll"; Components: arena
Type: files; Name: "{#Avenue}\SegcamMotionRust.dll"; Components: avenue

; windows-dlls holds the DirectX Shader Compiler with its licenses, which the FFGL DLLs and
; the .aex files load from their own folder; each OFX bundle carries its own copy.
; Extra Effects is shared with plugins from elsewhere, so these are counted and removed,
; without asking, with the last installer that put them there. The counts live in HKLM,
; which a TestRoot install can't write.
#ifdef TestRoot
  #define Shared ""
#else
  #define Shared " sharedfile uninsnosharedfileprompt"
#endif
[Files]
Source: "{#Source}\windows-dlls\*"; DestDir: "{#Arena}"; Components: arena; Flags: ignoreversion{#Shared}
Source: "{#Source}\windows-dlls\*"; DestDir: "{#Avenue}"; Components: avenue; Flags: ignoreversion{#Shared}

[Components]
Name: "examples"; Description: "Example effects from vfx-core"

#include Effects

[Code]
const
  NotFound = ' (not found)';

// True when a folder in Program Files matches Pattern, such as 'Nuke*'.
function ProgramFolderExists(Pattern: String): Boolean;
var
  Found: TFindRec;
begin
  Result := False;
  if FindFirst(ExpandConstant('{commonpf64}\') + Pattern, Found) then
  try
    repeat
      if Found.Attributes and FILE_ATTRIBUTE_DIRECTORY <> 0 then
        Result := True;
    until Result or not FindNext(Found);
  finally
    FindClose(Found);
  end;
end;

function OfxAppFound: Boolean;
begin
  Result := FileExists(ExpandConstant('{commonpf64}\Blackmagic Design\DaVinci Resolve\Resolve.exe'))
    or ProgramFolderExists('Nuke*') or ProgramFolderExists('Natron*')
    or ProgramFolderExists('INRIA\Natron*');
end;

function AdobeAppFound: Boolean;
begin
  Result := RegKeyExists(HKLM64, 'SOFTWARE\Adobe\After Effects')
    or RegKeyExists(HKLM64, 'SOFTWARE\Adobe\Premiere Pro');
end;

// App is 'Resolume Arena' or 'Resolume Avenue', found by its program folder: v0.1.1 made
// a Documents folder for Avenue on PCs without it. Resolume makes its Documents folder on
// first start, so one that hasn't started yet finds the plugins there when it does.
function ResolumeFound(App: String): Boolean;
begin
  Result := ProgramFolderExists(App);
end;

// The components an earlier install of this AppId chose, or '' for a first install.
function PreviousComponents: String;
var
  Key: String;
begin
  Result := '';
  Key := 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{{#AppGuid}}_is1';
  if IsAdminInstallMode then
    RegQueryStringValue(HKLM64, Key, 'Inno Setup: Selected Components', Result)
  else
    RegQueryStringValue(HKCU, Key, 'Inno Setup: Selected Components', Result);
end;

// The list given with /COMPONENTS= on the command line, or ''.
function CommandLineComponents: String;
var
  I: Integer;
begin
  Result := '';
  for I := 1 to ParamCount do
    if Pos('/COMPONENTS=', Uppercase(ParamStr(I))) = 1 then
      Result := Copy(ParamStr(I), Length('/COMPONENTS=') + 1, MaxInt);
end;

// Apps, a component list from v0.1.1 or later, in this Setup's names. v0.1.1 had a
// component for the lead effect itself, and one for both Resolumes, which now means the
// Resolumes installed.
function CurrentNames(Apps: String): String;
begin
  Result := ',' + Lowercase(Apps) + ',';
  StringChangeEx(Result, ',{#LowerCase(LeadFile)},', ',', True);
  if Pos(',ffgl,', Result) > 0 then
  begin
    StringChangeEx(Result, ',ffgl,', ',', True);
    if ResolumeFound('Resolume Arena') then Result := Result + 'arena,';
    if ResolumeFound('Resolume Avenue') then Result := Result + 'avenue,';
  end;
  while (Result <> '') and (Result[1] = ',') do
    Delete(Result, 1, 1);
  while (Result <> '') and (Result[Length(Result)] = ',') do
    Delete(Result, Length(Result), 1);
end;

function AppChosen: Boolean;
begin
  Result := WizardIsComponentSelected('ofx') or WizardIsComponentSelected('arena')
    or WizardIsComponentSelected('avenue') or WizardIsComponentSelected('adobe');
end;

procedure MarkNotFound(Caption: String);
var
  List: TNewCheckListBox;
  I: Integer;
begin
  List := WizardForm.ComponentsList;
  for I := 0 to List.Items.Count - 1 do
    if List.ItemCaption[I] = Caption then
      List.ItemCaption[I] := Caption + NotFound;
end;

// The apps Setup found, as a WizardSelectComponents list.
function FoundApps: String;
begin
  Result := '!ofx,!arena,!avenue,!adobe';
  if OfxAppFound then StringChangeEx(Result, '!ofx', 'ofx', True);
  if ResolumeFound('Resolume Arena') then StringChangeEx(Result, '!arena', 'arena', True);
  if ResolumeFound('Resolume Avenue') then StringChangeEx(Result, '!avenue', 'avenue', True);
  if AdobeAppFound then StringChangeEx(Result, '!adobe', 'adobe', True);
end;

var
  AppsPreselected: Boolean;

// Preselects what the command line or an earlier install chose, or else the apps Setup
// found, once. Inno sets the list from the type after InitializeWizard, restores an earlier
// choice only when its type still exists (v0.1.1's recommended and full are gone), and skips
// names it doesn't know. So this runs when the page first shows, or in a silent install.
procedure PreselectApps;
var
  Apps: String;
begin
  if AppsPreselected then
    Exit;
  AppsPreselected := True;
  Apps := CommandLineComponents;
  if Apps = '' then
  begin
    Apps := CurrentNames(PreviousComponents);
    if Apps = '' then
      Apps := FoundApps;
  end
  else
    Apps := CurrentNames(Apps);
  if Apps <> '' then
    WizardSelectComponents(Apps);
end;

// Where the effect shows up in each chosen app, one line each.
function WhereToFind: String;
begin
  Result := '';
  if WizardIsComponentSelected('ofx') then
    Result := Result + #13#10 + 'Resolve: Effects, Open FX, Filters, {#Category}';
  if WizardIsComponentSelected('arena') or WizardIsComponentSelected('avenue') then
    Result := Result + #13#10 + 'Resolume: Effects, {#AppName}';
  if WizardIsComponentSelected('adobe') then
    Result := Result + #13#10 + 'After Effects: Effect, {#Category}';
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  if CurPageID = wpSelectComponents then
    PreselectApps;
  if CurPageID = wpFinished then
    WizardForm.FinishedLabel.Caption :=
      '{#AppName} is installed. Restart any open apps, then find it here:' + #13#10 + WhereToFind;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  if WizardSilent then
    PreselectApps;
  Result := True;
  if (CurPageID = wpSelectComponents) and not WizardSilent and not AppChosen then
  begin
    MsgBox('Check at least one app.', mbError, MB_OK);
    Result := False;
  end;
end;

// A silent install with no app found or given stops here, with exit code 7.
function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  Result := '';
  if not AppChosen then
    Result := 'No app to add {#AppName} to: none was found or given.';
end;

procedure InitializeWizard;
begin
  if not OfxAppFound then MarkNotFound('DaVinci Resolve, Nuke and Natron (OpenFX)');
  if not ResolumeFound('Resolume Arena') then MarkNotFound('Resolume Arena');
  if not ResolumeFound('Resolume Avenue') then MarkNotFound('Resolume Avenue');
  if not AdobeAppFound then MarkNotFound('After Effects and Premiere Pro');
end;

// The ready page lists the folders each chosen app gets.
function UpdateReadyMemo(Space, NewLine, MemoUserInfoInfo, MemoDirInfo, MemoTypeInfo,
  MemoComponentsInfo, MemoGroupInfo, MemoTasksInfo: String): String;
begin
  Result := MemoComponentsInfo + NewLine + NewLine + 'Folders:';
  if WizardIsComponentSelected('ofx') then
    Result := Result + NewLine + Space + ExpandConstant('{#OfxPlugins}');
  if WizardIsComponentSelected('arena') then
    Result := Result + NewLine + Space + ExpandConstant('{#Arena}');
  if WizardIsComponentSelected('avenue') then
    Result := Result + NewLine + Space + ExpandConstant('{#Avenue}');
  if WizardIsComponentSelected('adobe') then
    Result := Result + NewLine + Space + ExpandConstant('{#MediaCore}');
end;
