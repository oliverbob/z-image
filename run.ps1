$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
python "$RootDir/run.py" $args
