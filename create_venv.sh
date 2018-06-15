#!/bin/bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

virtualenv --system-site-packages -p python2.7 $MYDIR/venv
source $MYDIR/venv/bin/activate
pip2.7 install --upgrade -r $MYDIR/requirements-devel.txt
pip2.7 install -e $MYDIR

