	??? ??%@??? ??%@!??? ??%@	L?b,???L?b,???!L?b,???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??? ??%@{?G?z??A??C?l%@Y?v??/??*	      I@2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCache????????!      I@)?I+???1      F@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch9??v????!      :@)9??v????1      :@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism/?$???!      E@)????Mb??1      0@:Preprocessing2F
Iterator::Model????????!      I@)????Mbp?1       @:Preprocessing2x
AIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl?~j?t?h?!      @)?~j?t?h?1      @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9L?b,???I?:??_?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	{?G?z??{?G?z??!{?G?z??      ??!       "      ??!       *      ??!       2	??C?l%@??C?l%@!??C?l%@:      ??!       B      ??!       J	?v??/???v??/??!?v??/??R      ??!       Z	?v??/???v??/??!?v??/??b      ??!       JCPU_ONLYYL?b,???b q?:??_?X@