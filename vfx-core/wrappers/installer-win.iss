; The Windows installer for the vfx-core plugins: OpenFX (Resolve, Nuke, Natron), FFGL
; (Resolume) and After Effects / Premiere Pro, for one effect with the others as options.
; package-win.sh builds the plugins, writes Effects (an .iss that includes
; installer-effect-win.iss once per effect) and runs this with the lead effect's names:
;   ISCC -DVersion=1.2.3 -DSource=<built plugins> -DEffects=<effects.iss> -DAppName=...
;        -DPublisher=... -DAppGuid=... installer-win.iss
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
WizardStyle=modern
Compression=lzma2
SolidCompression=yes
; Hosts load .ofx and .aex files too, so those count as files in use.
CloseApplicationsFilter=*.exe,*.dll,*.chm,*.ofx,*.aex

[Types]
Name: "recommended"; Description: "{#AppName} for Resolve and Resolume"
Name: "full"; Description: "Every effect for every app"
Name: "custom"; Description: "Choose apps and effects"; Flags: iscustom

; A plugin is installed when both its app and its effect are chosen.
[Components]
Name: "ofx"; Description: "DaVinci Resolve, Nuke, Natron (OpenFX)"; Types: recommended full
Name: "ffgl"; Description: "Resolume Avenue and Arena (FFGL)"; Types: recommended full
Name: "adobe"; Description: "After Effects and Premiere Pro"; Types: full

; v0.1.0 installed segcam as SegcamMotionRust. Its files go for each app chosen, so no app
; lists segcam twice. The binaries are named on their own so that Setup's check for files in
; use sees them, and closes an app that has one loaded.
[InstallDelete]
Type: files; Name: "{#OfxPlugins}\SegcamMotionRust.ofx.bundle\Contents\Win64\SegcamMotionRust.ofx"; Components: ofx
Type: filesandordirs; Name: "{#OfxPlugins}\SegcamMotionRust.ofx.bundle"; Components: ofx
Type: files; Name: "{#MediaCore}\SegcamMotionRust\SegcamMotionRust.aex"; Components: adobe
Type: filesandordirs; Name: "{#MediaCore}\SegcamMotionRust"; Components: adobe
Type: files; Name: "{#Documents}\Resolume Avenue\Extra Effects\SegcamMotionRust.dll"; Components: ffgl
Type: files; Name: "{#Documents}\Resolume Arena\Extra Effects\SegcamMotionRust.dll"; Components: ffgl

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
Source: "{#Source}\windows-dlls\*"; DestDir: "{#Documents}\Resolume Avenue\Extra Effects"; Components: ffgl; Check: ForAvenue; Flags: ignoreversion{#Shared}
Source: "{#Source}\windows-dlls\*"; DestDir: "{#Documents}\Resolume Arena\Extra Effects"; Components: ffgl; Check: ForArena; Flags: ignoreversion{#Shared}

#include Effects

[Code]
// Resolume makes its Documents folder on first start. Arena gets the plugins when its
// folder is there; Avenue's is the default, so a later Avenue install finds them.
function ForArena: Boolean;
begin
  Result := DirExists(ExpandConstant('{#Documents}\Resolume Arena'));
end;

function ForAvenue: Boolean;
begin
  Result := DirExists(ExpandConstant('{#Documents}\Resolume Avenue')) or not ForArena;
end;
