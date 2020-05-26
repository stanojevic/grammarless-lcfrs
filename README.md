Span-Based LCFRS-2 Parsing
=========

This is the implementation of the parser published in IWPT-2020:

Span-Based LCFRS-2 Parsing \
Miloš Stanojević and Mark Steedman \
University of Edinburgh 

Installation
---------------

Basic requirements:
- g++ with c++11 support (gcc4.8.1+) and pthreads
- cmake
- JDK 8 (Scala requires JDK >=8 but DyNet at the moment has problem with JDK >=9) you MUST set JAVA_HOME variable
- git
- mercurial

If all the basic requirements (including seeting the JAVA_HOME shell variable) are installed the rest
of the requirements will be installed automatically by first going to the root directory of the project
(the one that contains "src" as subdirectory) and run the following command:

     ./scripts/install_dependencies.sh

This command will take some time to install all the other dependencies (Scala, SBT, SWIG, Jep, Eigen and DyNet) and store them in directories `dependencies` and `lib`.

Parsing
---------

To run the parser execute the following command:

    $CODE_DIR/scripts/run.sh edin.mcfg.MainParse \
    --model-dir $MODEL_DIR \
    --words-file $TEST_WORDS \
    --tags-file $TEST_TAGS \
    --rules-to-use MCFG \
    --output-file $PRED_TREES

For $MODEL_DIR you can uncompress one of the models in the `models` directory.

$TEST_TAGS also have to be provided by the user (the parser doesn't do the tagging step).

$PRED_TREES will be outputed in the export format.


