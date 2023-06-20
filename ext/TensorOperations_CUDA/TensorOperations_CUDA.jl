module TensorOperations_CUDA
using cuTENSOR, CUDA

if CUDA.functional() && cuTENSOR.has_cutensor()
    const CuArray = CUDA.CuArray
    const CublasFloat = CUDA.CUBLAS.CublasFloat
    const CublasReal = CUDA.CUBLAS.CublasReal
    for s in (:handle, :CuTensorDescriptor, :cudaDataType_t,
            :cutensorContractionDescriptor_t, :cutensorContractionFind_t,
            :cutensorContractionPlan_t,
            :CUTENSOR_OP_IDENTITY, :CUTENSOR_OP_CONJ, :CUTENSOR_OP_ADD,
            :CUTENSOR_ALGO_DEFAULT,  :CUTENSOR_WORKSPACE_RECOMMENDED,
            :cutensorPermutation, :cutensorElementwiseBinary, :cutensorReduction,
            :cutensorReductionGetWorkspace, :cutensorComputeType,
            :cutensorGetAlignmentRequirement, :cutensorInitContractionDescriptor,
            :cutensorInitContractionFind, :cutensorContractionGetWorkspace,
            :cutensorInitContractionPlan, :cutensorContraction)
        eval(:(const $s = cuTENSOR.$s))
    end
    if isdefined(CUDA, :default_stream)
        const default_stream = CUDA.default_stream
    else
        const default_stream = CUDA.CuDefaultStream
    end
    include("implementation/cuarray.jl")
    @nospecialize
    include("indexnotation/cutensormacros.jl")
    @specialize
end

end
