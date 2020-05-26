#!/bin/bash

realpath() { [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}" ; }
PROJECT_DIR=$(dirname $(dirname $(realpath "$0")))
LIB_DIR=${PROJECT_DIR}/lib

source ${LIB_DIR}/SHELL_VARS

# setting up dynamic .so libraries
ALL_SO=${LIBPYTHON_HOME}:${PROJECT_DIR}/lib
export LD_LIBRARY_PATH=${ALL_SO}:${LD_LIBRARY_PATH}
export DYLD_LIBRARY_PATH=${ALL_SO}:${DYLD_LIBRARY_PATH}
export DYLD_INSERT_LIBRARIES=${ALL_SO}:${DYLD_INSERT_LIBRARIES}

# run the thing
exec java \
  ${DEBUG_REMOTELY} \
  -Djava.library.path=${LIB_DIR} \
  -Xmx${JVM_MEM} \
  -cp ${PROJECT_DIR}/target/scala-2.12/*assembly*.jar \
  $@

