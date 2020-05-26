#!/bin/bash

# PREREQUISITES
# - g++ with c++11 support (gcc4.8.1+) and pthreads -- you probably have this if you use non-windows machine made in this centuary
# - cmake
# - JDK 8 (Scala requires JDK >=8 but, I think, DyNet at the moment has problem with JDK >=9) you MUST set JAVA_HOME variable
# - git
# - python3 and python3-dev with numpy

# OPTIONAL
# - Intel MKL                           -- speeds up computation on Intel  CPU (you need to set USE_MKL=1 MKL_ROOT=...)
# - CUDA and cuDNN                      -- speeds up computation on NVidia GPU (you need to set USE_CUDA=1 CUDNN_ROOT=... CUDA_TOOLKIT_ROOT_DIR=...)
# - pip3 install allennlp               -- if you want to use ELMo embeddings
# - pip3 install pytorch-transformers   -- if you want to use BERT embeddings

set -e # exit on error

START_DIR=$PWD
SCALA_VER=2.12.10

if [[ -z "$CORES" ]]; then
  CORES=8   # Number of CPU cores to use during compilation
fi

realpath() { [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}" ; }

if [[ $(basename $(dirname $(realpath "$0"))) == "scripts" ]]; then
  PROJECT_DIR=$(dirname $(dirname $(realpath $0)))
elif [[ -e "$PWD/scripts" ]]; then
  PROJECT_DIR=$PWD
else
  echo you are probably running this script from a wrong directory
  exit 1
fi

echo PROJECT DIR IS ${PROJECT_DIR}

DEPENDENCIES_DIR=${PROJECT_DIR}/dependencies
LIB_DIR=${PROJECT_DIR}/lib

if [[ -e "${LIB_DIR}" || -e "${DEPENDENCIES_DIR}" || -e "${PROJECT_DIR}/target" ]] ; then
  echo please delete $(basename ${LIB_DIR}), $(basename ${DEPENDENCIES_DIR}) and $(basename ${PROJECT_DIR}/target) dirs before running the installation again
  echo you can run:   rm -rf ${LIB_DIR} ${DEPENDENCIES_DIR} ${PROJECT_DIR}/target
  exit 1
fi

SHELL_VARS_FILE=${LIB_DIR}/SHELL_VARS
mkdir -p ${LIB_DIR}
rm -f ${SHELL_VARS_FILE}
echo "# export DEBUG_REMOTELY=\"-Xdebug -Xrunjdwp:transport=dt_socket,address=8000,server=y,suspend=y\"" >> ${SHELL_VARS_FILE}
echo "# export DEBUG_DYNET=1"                                                                            >> ${SHELL_VARS_FILE}
echo "export JVM_MEM=6G        # how much memory to dedicate to JVM"                                     >> ${SHELL_VARS_FILE}
# chmod +x ${SHELL_VARS_FILE}

##########################    CUDA    #########################

if [[ ! -z "${USE_CUDA}" && "${USE_CUDA}"=="1" ]] ; then
  echo "compiling with GPU support"
  export CUDNN_ROOT="/opt/cuDNN-5.1_7.5" # this DyNet version can use only the older CUDNN
  export CUDA_TOOLKIT_ROOT_DIR="/opt/cuda-10.1.168_418_67"
  CUDA="-DBACKEND=cuda -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_TOOLKIT_ROOT_DIR -DCUDNN_ROOT=$CUDNN_ROOT" # this is to use GPU
else
  echo "compiling only for CPU support"
  CUDA="-DBACKEND=eigen" # no CUDA
fi

##########################  Intel MKL #########################

if [[ ! -z "${USE_MKL}" && "${USE_MKL}"=="1" ]] ; then
  echo "compiling with MKL support"
  if [[ -z "${MKL_ROOT}" ]] ; then
    MKL_ROOT=`ls -1d $HOME/intel/compilers_and_libraries_20*.*/linux/mkl/ | head -1`
  fi
  [[ ! -z "$MKL_ROOT" ]] || { echo 'finding Intel MKL failed' ; exit 1; }
  echo "FOUND MKL AT ${MKL_ROOT}"
  MKL="-DMKL=TRUE -DMKL_ROOT=${MKL_ROOT}"

  echo "export MKL_NUM_THREADS=4" >> ${SHELL_VARS_FILE}
else
  echo "compiling with NO MKL"
fi

##########################  SPEED vs PORTABILITY #########################

if [[ ! -z "${OPTIMIZE_PORTABILITY}" && "${OPTIMIZE_PORTABILITY}"=="1" ]] ; then
  if [[ ! -z "${USE_MKL}" && "${USE_MKL}"=="1" ]] ; then
    echo "you can't use MKL and optimize for portability"
    exit -1
  fi
  if [[ ! -z "${USE_CUDA}" && "${USE_CUDA}"=="1" ]] ; then
    echo "you can't use CUDA and optimize for portability"
    exit -1
  fi
  echo "optimizing for PORTABILITY"
  DYNET_OPT_LEVEL=3
  DYNET_ARCHITECTURE=x86-64
else
  echo "optimizing for SPEED"
  DYNET_OPT_LEVEL=fast
  DYNET_ARCHITECTURE=native
fi

########################  installation ########################
echo "using ${CORES} cores during compile time"
echo
echo
echo

mkdir -p ${DEPENDENCIES_DIR}
cd ${DEPENDENCIES_DIR}

##########################    installing sbt    #########################

SBT_VERSION=1.2.8
SBT_DIR=${DEPENDENCIES_DIR}/sbt
rm -rf ${SBT_DIR}
wget --no-check-certificate https://piccolo.link/sbt-${SBT_VERSION}.tgz || { echo 'SBT download failed' ; exit 1; }
tar xfvz sbt-${SBT_VERSION}.tgz
rm sbt-${SBT_VERSION}.tgz
export PATH=${SBT_DIR}/bin:$PATH

##########################    installing swig    #########################

SWIG_DIR=${DEPENDENCIES_DIR}/swig
wget --no-check-certificate 'https://downloads.sourceforge.net/project/swig/swig/swig-3.0.12/swig-3.0.12.tar.gz?r=https%3A%2F%2Fsourceforge.net%2Fprojects%2Fswig%2Ffiles%2Flatest%2Fdownload&ts=1529975837' -O swig-3.0.12.tar.gz
tar xfvz swig-3.0.12.tar.gz
rm swig-3.0.12.tar.gz
cd swig-3.0.12
./configure --without-pcre --prefix=${SWIG_DIR} || { echo 'SWIG install failed' ; exit 1; }
make -j ${CORES}                                || { echo 'SWIG install failed' ; exit 1; }
make install                                    || { echo 'SWIG install failed' ; exit 1; }
cd ..
rm -rf swig-3.0.12
export PATH=${SWIG_DIR}/bin:$PATH

##########################    installing Jep    #########################
PYTHON_VER=3
JEP_DIR=${LIB_DIR}/jep
mkdir -p ${LIB_DIR}
rm -rf jep_tmp_gitclone jep
git clone https://github.com/ninia/jep.git jep_tmp_gitclone
cd jep_tmp_gitclone
# git checkout 05ec104d7aa77e1cbd80750372b8abc8dbc6f3c4 # version 3.6.4  # works with python2
# git checkout 7fc93a6319de8503d8a50e75afdeb30ad19e6fac # version 3.7.1
git checkout b94909e4b02f18e45f4cccdd73cea19caad0dfe9 # version 3.8
python${PYTHON_VER} setup.py install --prefix=temporary || { echo 'Jep install failed' ; exit 1; }
mv temporary/*/*/*/*/*.jar ${LIB_DIR}
mv temporary/lib*/python*/site-packages/jep ${JEP_DIR}

cp ${JEP_DIR}/jep.*.so      ${LIB_DIR}/libjep.so
if [[ -e "${JEP_DIR}/libjep.jnilib" ]] ; then
    cp ${JEP_DIR}/libjep.jnilib ${LIB_DIR}/libjep.jnilib
fi
cd ${JEP_DIR}
rm -rf libjep* jep.cpython* __pycache__
cd -
cd ..
rm -rf jep_tmp_gitclone

PYTHON_LIBS_DIR=$(python${PYTHON_VER} -c "import distutils.sysconfig,string; print(' '.join(filter(None,distutils.sysconfig.get_config_vars('LIBDIR'   ))))")
PYTHON_SO_NAME=$( python${PYTHON_VER} -c "import distutils.sysconfig,string; print(' '.join(filter(None,distutils.sysconfig.get_config_vars('LDLIBRARY'))))")
PYTHON_SO_PATH=$(find ${PYTHON_LIBS_DIR} -name ${PYTHON_SO_NAME} | head -n 1 | xargs readlink -f)
echo "export LIBPYTHON_HOME=$(dirname ${PYTHON_SO_PATH})" >> ${SHELL_VARS_FILE}

##########################    installing DyNet    #########################

EIGEN_COMMIT=b2e267dc99d4
wget --no-check-certificate https://bitbucket.org/eigen/eigen/get/${EIGEN_COMMIT}.zip || { echo 'Eigen download failed' ; exit 1; }
unzip ${EIGEN_COMMIT}.zip                                                             || { echo 'Eigen download failed' ; exit 1; }
rm    ${EIGEN_COMMIT}.zip                                                             || { echo 'Eigen download failed' ; exit 1; }
mv eigen-eigen-${EIGEN_COMMIT} eigen                                                  || { echo 'Eigen download failed' ; exit 1; }


# original DyNet
git clone https://github.com/clab/dynet.git                                           || { echo 'DyNet download failed' ; exit 1; }
cd dynet                                                                              || { echo 'DyNet download failed' ; exit 1; }
DYNET_COMMIT=a581587cddd25c295f5b23339bb537d32dca8ee1                                 || { echo 'DyNet download failed' ; exit 1; }
git checkout ${DYNET_COMMIT}                                                          || { echo 'DyNet download failed' ; exit 1; }
cd ..                                                                                 || { echo 'DyNet download failed' ; exit 1; }

# swig-multi-device DyNet
git clone https://github.com/shuheik/dynet.git BETTER_SWIG                            || { echo 'DyNet BETTER_SWIG failed' ; exit 1; }
cd BETTER_SWIG                                                                        || { echo 'DyNet BETTER_SWIG failed' ; exit 1; }
git checkout 429d598b5b8295aa756f652c8e8b4a357450f532                                 || { echo 'DyNet BETTER_SWIG failed' ; exit 1; }
cd ..                                                                                 || { echo 'DyNet BETTER_SWIG failed' ; exit 1; }
rm -rf ./dynet/contrib/swig                                                           || { echo 'DyNet BETTER_SWIG failed' ; exit 1; }
cp -r  BETTER_SWIG/contrib/swig ./dynet/contrib/swig                                  || { echo 'DyNet BETTER_SWIG failed' ; exit 1; }
rm -rf BETTER_SWIG                                                                    || { echo 'DyNet BETTER_SWIG failed' ; exit 1; }

# getting improved version of matrix multiplication for CPU when Intel MKL is installed
git clone https://github.com/artidoro/dynet.git MKL_MATMUL                            || { echo 'DyNet MKL_MATMUL_BATCH failed' ; exit 1; }
cd MKL_MATMUL                                                                         || { echo 'DyNet MKL_MATMUL_BATCH failed' ; exit 1; }
git checkout e564789f2bd437b7c2398cfe6dde1473c33cd5cf                                 || { echo 'DyNet MKL_MATMUL_BATCH failed' ; exit 1; }
cd ..                                                                                 || { echo 'DyNet MKL_MATMUL_BATCH failed' ; exit 1; }
cp MKL_MATMUL/dynet/matrix-multiply.h ./dynet/dynet/matrix-multiply.h                 || { echo 'DyNet MKL_MATMUL_BATCH failed' ; exit 1; }
rm -rf MKL_MATMUL                                                                     || { echo 'DyNet MKL_MATMUL_BATCH failed' ; exit 1; }

cd dynet

# this is for more flexible transfer of binaries across computer architectures
perl -pi -e "s/march=native/march=${DYNET_ARCHITECTURE}/" CMakeLists.txt

# tests of Scala usually fail because of the memory leaking (don't worry about it)
rm -rf contrib/swig/src/test

# Make DyNet allow for immediate computation and checking for NaN -- useful for debugging and not available in standard Scala bindings
perl -pi -e 's/~ComputationGraph\(\);/$&\n  void set_immediate_compute(bool b);/'                            contrib/swig/dynet_swig.i
perl -pi -e 's/~ComputationGraph\(\);/$&\n  void set_check_validity(bool b);/'                               contrib/swig/dynet_swig.i
perl -pi -e 's/ *var version/  def setImmediateCompute(b:Boolean) : Unit = cg.set_immediate_compute(b)\n$&/' contrib/swig/src/main/scala/edu/cmu/dynet/ComputationGraph.scala
perl -pi -e 's/ *var version/  def setCheckValidity(   b:Boolean) : Unit = cg.set_check_validity(b)\n$&/'    contrib/swig/src/main/scala/edu/cmu/dynet/ComputationGraph.scala

mkdir -p build
cd build
cmake .. -DRELEASE_OPT_LEVEL=${DYNET_OPT_LEVEL} -DEIGEN3_INCLUDE_DIR=${DEPENDENCIES_DIR}/eigen -DENABLE_SWIG=ON -DSCALA_VERSION=${SCALA_VER} ${CUDA} ${MKL} || { echo 'DyNet install failed' ; exit 1; }

make -j ${CORES} || { echo 'DyNet install failed' ; exit 1; }

if [[ ! -z "${OPTIMIZE_PORTABILITY}" && "${OPTIMIZE_PORTABILITY}"=="1" ]] ; then
  if [[ "$(uname -s)"=="Linux" ]]; then
     # linux
     c++ -fPIC -DEIGEN_FAST_MATH -fno-finite-math-only -Wall -Wno-missing-braces -std=c++11 -g -funroll-loops -Ofast -march=x86-64 -shared     -Wl,-soname,libdynet.so,-z,muldefs -o libdynet.so $(find -name \*.o | tr "\n" " ") -lpthread || { echo 'DyNet install failed' ; exit 1; }
  else
     # mac
     c++ -fPIC -DEIGEN_FAST_MATH -fno-finite-math-only -Wall -Wno-missing-braces -std=c++11 -g -funroll-loops -Ofast -march=x86-64 -dynamiclib -Wl,-headerpad_max_install_names -o libdynet.jnilib -install_name libdynet.dylib $(find -name \*.o | tr "\n" " ") || { echo 'DyNet install failed' ; exit 1; }
  fi
  strip --strip-unneeded libdynet.* || { echo 'DyNet install failed' ; exit 1; }
  mv libdynet.* dynet/              || { echo 'DyNet install failed' ; exit 1; }
fi

mkdir -p ${LIB_DIR}
cd ${LIB_DIR}

# rm -f dynet*.jar libdynet*

# mv ${DEPENDENCIES_DIR}/dynet/build/contrib/swig/dynet_swigJNI.jar .      # java lib
mv ${DEPENDENCIES_DIR}/dynet/build/contrib/swig/dynet_swigJNI_dylib.jar  . # native lib
mv ${DEPENDENCIES_DIR}/dynet/build/contrib/swig/dynet_swigJNI_scala*.jar . # scala lib
cp ${DEPENDENCIES_DIR}/dynet/build/dynet/libdynet.*                      . # dynet *.so

##########################    installing EasyCCG    #########################

cd ${LIB_DIR}
wget --no-check-certificate https://github.com/mikelewis0/easyccg/raw/master/easyccg.jar

#########################################################################

cd ${PROJECT_DIR}

OPTIMIZED=true sbt clean assembly || { echo 'sbt assembly failed' ; exit 1; }

rm -rf ${DEPENDENCIES_DIR}/swig
rm -rf ${DEPENDENCIES_DIR}/eigen
rm -rf ${DEPENDENCIES_DIR}/dynet
rm -rf ${DEPENDENCIES_DIR}/sbt
rmdir ${DEPENDENCIES_DIR}

cd ${START_DIR}

