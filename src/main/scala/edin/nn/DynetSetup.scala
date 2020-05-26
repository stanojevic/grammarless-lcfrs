package edin.nn

import edu.cmu.dynet.internal.{Device, DynetParams}
import edu.cmu.dynet.{ComputationGraph, DeviceManager, internal}

object DynetSetup {

  private var dynetIsInitialized = false

  def availableGPUs : List[Device] = {
    require(dynetIsInitialized)
    (0 until DeviceManager.numDevices().toInt).map(DeviceManager.get(_)).filter(_.getName startsWith "GPU:").toList
  }
  var currentDevice : Device = _

  private var seedVal : Long = 2526024616L //  System.currentTimeMillis()
  def seed : Long = seedVal

  def init_dynet(dynet_mem:String=null, autobatch:Int=0, seed:Long=this.seed, gpus:Int=0, devices:String=null) : Unit = {
    if(dynetIsInitialized)
      sys.error("you can't initialize DyNet twice")
    else
      dynetIsInitialized = true
    val params:DynetParams = new internal.DynetParams()

    if(seed != this.seed){
      seedVal = seed
      params.setRandom_seed(seed)
    }

    params.setAutobatch(autobatch)
    if(dynet_mem != null)
      params.setMem_descriptor(dynet_mem)

    if(gpus>0)
      params.setRequested_gpus(gpus)
    if(devices != null){
      require(!params.getIds_requested)
      params.setIds_requested(true)
      for(s <- devices split ','){
        require(s startsWith "GPU:")
        val gpuID = s.split(":")(1).toInt
        params.getGpu_mask.set(gpuID, params.getGpu_mask.get(gpuID) + 1)
        params.setRequested_gpus(params.getRequested_gpus + 1)
        require(params.getGpu_mask.get(gpuID) == 1)
      }
    }

    internal.dynet_swig.initialize(params)

    currentDevice = DeviceManager.getDefaultDevice

    if(System.getenv("DEBUG_DYNET") == "1")
      startDebuggingMode()

  }

  private var expressions = List[AnyRef]()
  def safeReference(x:AnyRef) : Unit = {
    expressions ::= x
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

  private var cg_counter = 0
  def cg_id : Int = cg_counter

  def cg_renew() : Unit = {
    expressions = List()
    cg_counter += 1
    ComputationGraph.renew()
    if(isDebugging){
      ComputationGraph.setImmediateCompute(true)
      ComputationGraph.setCheckValidity(true)
    }
  }

  def startDebuggingMode() : Unit = {
    DEBUGGING_MODE = true
    System.err.println("[dynet] ENTERING DEBUGGING MODE")
    ComputationGraph.setImmediateCompute(true)
    ComputationGraph.setCheckValidity(true)
  }
  def isDebugging : Boolean = DEBUGGING_MODE
  private var DEBUGGING_MODE: Boolean = false

}
