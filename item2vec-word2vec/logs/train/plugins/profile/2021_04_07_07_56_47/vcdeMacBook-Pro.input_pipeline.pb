	?ʡE??%@?ʡE??%@!?ʡE??%@	Q#m??"??Q#m??"??!Q#m??"??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?ʡE??%@???x?&??A?O??nR%@Y?rh??|??*	     :?@2?
`Iterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::TensorSlice?}?5^?I@!?!?
?I@)}?5^?I@1?!?
?I@:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::BatchV2::Shuffle??(\???@!g??%YX@)V-??@1hC???G@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch????Mb??!J??????)????Mb??1J??????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism????????!Z??@??);?O??n??1?|???g??:Preprocessing2F
Iterator::Modelh??|?5??!????-??);?O??n??1?|???g??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9Q#m??"??IsK%t?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???x?&?????x?&??!???x?&??      ??!       "      ??!       *      ??!       2	?O??nR%@?O??nR%@!?O??nR%@:      ??!       B      ??!       J	?rh??|???rh??|??!?rh??|??R      ??!       Z	?rh??|???rh??|??!?rh??|??b      ??!       JCPU_ONLYYQ#m??"??b qsK%t?X@