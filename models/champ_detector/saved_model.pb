ű˙
Ý
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68Ő

conv2d_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_23/kernel
}
$conv2d_23/kernel/Read/ReadVariableOpReadVariableOpconv2d_23/kernel*&
_output_shapes
:@*
dtype0
t
conv2d_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_23/bias
m
"conv2d_23/bias/Read/ReadVariableOpReadVariableOpconv2d_23/bias*
_output_shapes
:@*
dtype0

conv2d_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_24/kernel
}
$conv2d_24/kernel/Read/ReadVariableOpReadVariableOpconv2d_24/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_24/bias
m
"conv2d_24/bias/Read/ReadVariableOpReadVariableOpconv2d_24/bias*
_output_shapes
:@*
dtype0

conv2d_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_25/kernel
}
$conv2d_25/kernel/Read/ReadVariableOpReadVariableOpconv2d_25/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_25/bias
m
"conv2d_25/bias/Read/ReadVariableOpReadVariableOpconv2d_25/bias*
_output_shapes
:@*
dtype0
z
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ô*
shared_namedense_6/kernel
s
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel* 
_output_shapes
:
ô*
dtype0
q
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ô*
shared_namedense_6/bias
j
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes	
:ô*
dtype0
z
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ôľ*
shared_namedense_7/kernel
s
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel* 
_output_shapes
:
ôľ*
dtype0
q
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ľ*
shared_namedense_7/bias
j
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes	
:ľ*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/conv2d_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_23/kernel/m

+Adam/conv2d_23/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_23/kernel/m*&
_output_shapes
:@*
dtype0

Adam/conv2d_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_23/bias/m
{
)Adam/conv2d_23/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_23/bias/m*
_output_shapes
:@*
dtype0

Adam/conv2d_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_24/kernel/m

+Adam/conv2d_24/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_24/kernel/m*&
_output_shapes
:@@*
dtype0

Adam/conv2d_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_24/bias/m
{
)Adam/conv2d_24/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_24/bias/m*
_output_shapes
:@*
dtype0

Adam/conv2d_25/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_25/kernel/m

+Adam/conv2d_25/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_25/kernel/m*&
_output_shapes
:@@*
dtype0

Adam/conv2d_25/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_25/bias/m
{
)Adam/conv2d_25/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_25/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ô*&
shared_nameAdam/dense_6/kernel/m

)Adam/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/m* 
_output_shapes
:
ô*
dtype0

Adam/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ô*$
shared_nameAdam/dense_6/bias/m
x
'Adam/dense_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/m*
_output_shapes	
:ô*
dtype0

Adam/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ôľ*&
shared_nameAdam/dense_7/kernel/m

)Adam/dense_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/m* 
_output_shapes
:
ôľ*
dtype0

Adam/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ľ*$
shared_nameAdam/dense_7/bias/m
x
'Adam/dense_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/m*
_output_shapes	
:ľ*
dtype0

Adam/conv2d_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_23/kernel/v

+Adam/conv2d_23/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_23/kernel/v*&
_output_shapes
:@*
dtype0

Adam/conv2d_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_23/bias/v
{
)Adam/conv2d_23/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_23/bias/v*
_output_shapes
:@*
dtype0

Adam/conv2d_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_24/kernel/v

+Adam/conv2d_24/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_24/kernel/v*&
_output_shapes
:@@*
dtype0

Adam/conv2d_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_24/bias/v
{
)Adam/conv2d_24/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_24/bias/v*
_output_shapes
:@*
dtype0

Adam/conv2d_25/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_25/kernel/v

+Adam/conv2d_25/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_25/kernel/v*&
_output_shapes
:@@*
dtype0

Adam/conv2d_25/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_25/bias/v
{
)Adam/conv2d_25/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_25/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ô*&
shared_nameAdam/dense_6/kernel/v

)Adam/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/v* 
_output_shapes
:
ô*
dtype0

Adam/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ô*$
shared_nameAdam/dense_6/bias/v
x
'Adam/dense_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/v*
_output_shapes	
:ô*
dtype0

Adam/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ôľ*&
shared_nameAdam/dense_7/kernel/v

)Adam/dense_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/v* 
_output_shapes
:
ôľ*
dtype0

Adam/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ľ*$
shared_nameAdam/dense_7/bias/v
x
'Adam/dense_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/v*
_output_shapes	
:ľ*
dtype0

NoOpNoOp
ÓM
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*M
valueMBM BúL
ś
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
Ś

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
Ś

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses*

(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses* 
Ś

.kernel
/bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses*

6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses* 
Ś

<kernel
=bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses*
Ś

Dkernel
Ebias
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses*

Liter

Mbeta_1

Nbeta_2
	Odecay
Plearning_ratemm m!m.m/m<m=mDmEmvv v!v.v/v<v=vDvEv*
J
0
1
 2
!3
.4
/5
<6
=7
D8
E9*
J
0
1
 2
!3
.4
/5
<6
=7
D8
E9*
* 
°
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Vserving_default* 
`Z
VARIABLE_VALUEconv2d_23/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_23/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEconv2d_24/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_24/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

 0
!1*

 0
!1*
* 

anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEconv2d_25/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_25/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

.0
/1*

.0
/1*
* 

knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses* 
* 
* 
^X
VARIABLE_VALUEdense_6/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_6/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

<0
=1*

<0
=1*
* 

unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEdense_7/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_7/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

D0
E1*

D0
E1*
* 

znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
<
0
1
2
3
4
5
6
7*

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

total

count
	variables
	keras_api*
M

total

count

_fn_kwargs
	variables
	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

	variables*
}
VARIABLE_VALUEAdam/conv2d_23/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_23/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/conv2d_24/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_24/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/conv2d_25/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_25/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_6/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_6/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_7/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_7/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/conv2d_23/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_23/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/conv2d_24/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_24/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/conv2d_25/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_25/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_6/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_6/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_7/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_7/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_input_11Placeholder*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*$
shape:˙˙˙˙˙˙˙˙˙
é
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_11conv2d_23/kernelconv2d_23/biasconv2d_24/kernelconv2d_24/biasconv2d_25/kernelconv2d_25/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_55941
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¤
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_23/kernel/Read/ReadVariableOp"conv2d_23/bias/Read/ReadVariableOp$conv2d_24/kernel/Read/ReadVariableOp"conv2d_24/bias/Read/ReadVariableOp$conv2d_25/kernel/Read/ReadVariableOp"conv2d_25/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv2d_23/kernel/m/Read/ReadVariableOp)Adam/conv2d_23/bias/m/Read/ReadVariableOp+Adam/conv2d_24/kernel/m/Read/ReadVariableOp)Adam/conv2d_24/bias/m/Read/ReadVariableOp+Adam/conv2d_25/kernel/m/Read/ReadVariableOp)Adam/conv2d_25/bias/m/Read/ReadVariableOp)Adam/dense_6/kernel/m/Read/ReadVariableOp'Adam/dense_6/bias/m/Read/ReadVariableOp)Adam/dense_7/kernel/m/Read/ReadVariableOp'Adam/dense_7/bias/m/Read/ReadVariableOp+Adam/conv2d_23/kernel/v/Read/ReadVariableOp)Adam/conv2d_23/bias/v/Read/ReadVariableOp+Adam/conv2d_24/kernel/v/Read/ReadVariableOp)Adam/conv2d_24/bias/v/Read/ReadVariableOp+Adam/conv2d_25/kernel/v/Read/ReadVariableOp)Adam/conv2d_25/bias/v/Read/ReadVariableOp)Adam/dense_6/kernel/v/Read/ReadVariableOp'Adam/dense_6/bias/v/Read/ReadVariableOp)Adam/dense_7/kernel/v/Read/ReadVariableOp'Adam/dense_7/bias/v/Read/ReadVariableOpConst*4
Tin-
+2)	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_56212

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_23/kernelconv2d_23/biasconv2d_24/kernelconv2d_24/biasconv2d_25/kernelconv2d_25/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_23/kernel/mAdam/conv2d_23/bias/mAdam/conv2d_24/kernel/mAdam/conv2d_24/bias/mAdam/conv2d_25/kernel/mAdam/conv2d_25/bias/mAdam/dense_6/kernel/mAdam/dense_6/bias/mAdam/dense_7/kernel/mAdam/dense_7/bias/mAdam/conv2d_23/kernel/vAdam/conv2d_23/bias/vAdam/conv2d_24/kernel/vAdam/conv2d_24/bias/vAdam/conv2d_25/kernel/vAdam/conv2d_25/bias/vAdam/dense_6/kernel/vAdam/dense_6/bias/vAdam/dense_7/kernel/vAdam/dense_7/bias/v*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_56339 
ă3

H__inference_sequential_12_layer_call_and_return_conditional_losses_55871

inputsB
(conv2d_23_conv2d_readvariableop_resource:@7
)conv2d_23_biasadd_readvariableop_resource:@B
(conv2d_24_conv2d_readvariableop_resource:@@7
)conv2d_24_biasadd_readvariableop_resource:@B
(conv2d_25_conv2d_readvariableop_resource:@@7
)conv2d_25_biasadd_readvariableop_resource:@:
&dense_6_matmul_readvariableop_resource:
ô6
'dense_6_biasadd_readvariableop_resource:	ô:
&dense_7_matmul_readvariableop_resource:
ôľ6
'dense_7_biasadd_readvariableop_resource:	ľ
identity˘ conv2d_23/BiasAdd/ReadVariableOp˘conv2d_23/Conv2D/ReadVariableOp˘ conv2d_24/BiasAdd/ReadVariableOp˘conv2d_24/Conv2D/ReadVariableOp˘ conv2d_25/BiasAdd/ReadVariableOp˘conv2d_25/Conv2D/ReadVariableOp˘dense_6/BiasAdd/ReadVariableOp˘dense_6/MatMul/ReadVariableOp˘dense_7/BiasAdd/ReadVariableOp˘dense_7/MatMul/ReadVariableOp
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0­
conv2d_23/Conv2DConv2Dinputs'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
paddingSAME*
strides

 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@l
conv2d_23/ReluReluconv2d_23/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@Ž
max_pooling2d_15/MaxPoolMaxPoolconv2d_23/Relu:activations:0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
ksize
*
paddingVALID*
strides

conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Č
conv2d_24/Conv2DConv2D!max_pooling2d_15/MaxPool:output:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
paddingSAME*
strides

 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@l
conv2d_24/ReluReluconv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@Ž
max_pooling2d_16/MaxPoolMaxPoolconv2d_24/Relu:activations:0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
ksize
*
paddingVALID*
strides

conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Č
conv2d_25/Conv2DConv2D!max_pooling2d_16/MaxPool:output:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
paddingSAME*
strides

 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@l
conv2d_25/ReluReluconv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@`
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙ 	  
flatten_4/ReshapeReshapeconv2d_25/Relu:activations:0flatten_4/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
ô*
dtype0
dense_6/MatMulMatMulflatten_4/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:ô*
dtype0
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ôa
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource* 
_output_shapes
:
ôľ*
dtype0
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľ
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:ľ*
dtype0
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľg
dense_7/SoftmaxSoftmaxdense_7/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľi
IdentityIdentitydense_7/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľ
NoOpNoOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp!^conv2d_25/BiasAdd/ReadVariableOp ^conv2d_25/Conv2D/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:˙˙˙˙˙˙˙˙˙: : : : : : : : : : 2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2D
 conv2d_25/BiasAdd/ReadVariableOp conv2d_25/BiasAdd/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ř$
ů
H__inference_sequential_12_layer_call_and_return_conditional_losses_55772
input_11)
conv2d_23_55743:@
conv2d_23_55745:@)
conv2d_24_55749:@@
conv2d_24_55751:@)
conv2d_25_55755:@@
conv2d_25_55757:@!
dense_6_55761:
ô
dense_6_55763:	ô!
dense_7_55766:
ôľ
dense_7_55768:	ľ
identity˘!conv2d_23/StatefulPartitionedCall˘!conv2d_24/StatefulPartitionedCall˘!conv2d_25/StatefulPartitionedCall˘dense_6/StatefulPartitionedCall˘dense_7/StatefulPartitionedCallű
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCallinput_11conv2d_23_55743conv2d_23_55745*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_23_layer_call_and_return_conditional_losses_55437ó
 max_pooling2d_15/PartitionedCallPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_55404
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_15/PartitionedCall:output:0conv2d_24_55749conv2d_24_55751*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_55455ó
 max_pooling2d_16/PartitionedCallPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_55416
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_16/PartitionedCall:output:0conv2d_25_55755conv2d_25_55757*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_25_layer_call_and_return_conditional_losses_55473Ţ
flatten_4/PartitionedCallPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_55485
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_6_55761dense_6_55763*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_55498
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_55766dense_7_55768*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_55515x
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľö
NoOpNoOp"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:˙˙˙˙˙˙˙˙˙: : : : : : : : : : 2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:Y U
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
input_11

g
K__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_56001

inputs
identity˘
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:r n
J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
¸
L
0__inference_max_pooling2d_15_layer_call_fn_55966

inputs
identityŮ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_55404
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:r n
J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs



#__inference_signature_wrapper_55941
input_11!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:
ô
	unknown_6:	ô
	unknown_7:
ôľ
	unknown_8:	ľ
identity˘StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinput_11unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_55395p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:˙˙˙˙˙˙˙˙˙: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
input_11
ă3

H__inference_sequential_12_layer_call_and_return_conditional_losses_55914

inputsB
(conv2d_23_conv2d_readvariableop_resource:@7
)conv2d_23_biasadd_readvariableop_resource:@B
(conv2d_24_conv2d_readvariableop_resource:@@7
)conv2d_24_biasadd_readvariableop_resource:@B
(conv2d_25_conv2d_readvariableop_resource:@@7
)conv2d_25_biasadd_readvariableop_resource:@:
&dense_6_matmul_readvariableop_resource:
ô6
'dense_6_biasadd_readvariableop_resource:	ô:
&dense_7_matmul_readvariableop_resource:
ôľ6
'dense_7_biasadd_readvariableop_resource:	ľ
identity˘ conv2d_23/BiasAdd/ReadVariableOp˘conv2d_23/Conv2D/ReadVariableOp˘ conv2d_24/BiasAdd/ReadVariableOp˘conv2d_24/Conv2D/ReadVariableOp˘ conv2d_25/BiasAdd/ReadVariableOp˘conv2d_25/Conv2D/ReadVariableOp˘dense_6/BiasAdd/ReadVariableOp˘dense_6/MatMul/ReadVariableOp˘dense_7/BiasAdd/ReadVariableOp˘dense_7/MatMul/ReadVariableOp
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0­
conv2d_23/Conv2DConv2Dinputs'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
paddingSAME*
strides

 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@l
conv2d_23/ReluReluconv2d_23/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@Ž
max_pooling2d_15/MaxPoolMaxPoolconv2d_23/Relu:activations:0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
ksize
*
paddingVALID*
strides

conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Č
conv2d_24/Conv2DConv2D!max_pooling2d_15/MaxPool:output:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
paddingSAME*
strides

 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@l
conv2d_24/ReluReluconv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@Ž
max_pooling2d_16/MaxPoolMaxPoolconv2d_24/Relu:activations:0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
ksize
*
paddingVALID*
strides

conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Č
conv2d_25/Conv2DConv2D!max_pooling2d_16/MaxPool:output:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
paddingSAME*
strides

 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@l
conv2d_25/ReluReluconv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@`
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙ 	  
flatten_4/ReshapeReshapeconv2d_25/Relu:activations:0flatten_4/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
ô*
dtype0
dense_6/MatMulMatMulflatten_4/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:ô*
dtype0
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ôa
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource* 
_output_shapes
:
ôľ*
dtype0
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľ
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:ľ*
dtype0
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľg
dense_7/SoftmaxSoftmaxdense_7/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľi
IdentityIdentitydense_7/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľ
NoOpNoOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp!^conv2d_25/BiasAdd/ReadVariableOp ^conv2d_25/Conv2D/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:˙˙˙˙˙˙˙˙˙: : : : : : : : : : 2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2D
 conv2d_25/BiasAdd/ReadVariableOp conv2d_25/BiasAdd/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ľ

ö
B__inference_dense_6_layer_call_and_return_conditional_losses_56052

inputs2
matmul_readvariableop_resource:
ô.
biasadd_readvariableop_resource:	ô
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ô*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ôs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ô*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ôQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ôb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ôw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ć
`
D__inference_flatten_4_layer_call_and_return_conditional_losses_55485

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙ 	  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:˙˙˙˙˙˙˙˙˙@:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
Ź

!__inference__traced_restore_56339
file_prefix;
!assignvariableop_conv2d_23_kernel:@/
!assignvariableop_1_conv2d_23_bias:@=
#assignvariableop_2_conv2d_24_kernel:@@/
!assignvariableop_3_conv2d_24_bias:@=
#assignvariableop_4_conv2d_25_kernel:@@/
!assignvariableop_5_conv2d_25_bias:@5
!assignvariableop_6_dense_6_kernel:
ô.
assignvariableop_7_dense_6_bias:	ô5
!assignvariableop_8_dense_7_kernel:
ôľ.
assignvariableop_9_dense_7_bias:	ľ'
assignvariableop_10_adam_iter:	 )
assignvariableop_11_adam_beta_1: )
assignvariableop_12_adam_beta_2: (
assignvariableop_13_adam_decay: 0
&assignvariableop_14_adam_learning_rate: #
assignvariableop_15_total: #
assignvariableop_16_count: %
assignvariableop_17_total_1: %
assignvariableop_18_count_1: E
+assignvariableop_19_adam_conv2d_23_kernel_m:@7
)assignvariableop_20_adam_conv2d_23_bias_m:@E
+assignvariableop_21_adam_conv2d_24_kernel_m:@@7
)assignvariableop_22_adam_conv2d_24_bias_m:@E
+assignvariableop_23_adam_conv2d_25_kernel_m:@@7
)assignvariableop_24_adam_conv2d_25_bias_m:@=
)assignvariableop_25_adam_dense_6_kernel_m:
ô6
'assignvariableop_26_adam_dense_6_bias_m:	ô=
)assignvariableop_27_adam_dense_7_kernel_m:
ôľ6
'assignvariableop_28_adam_dense_7_bias_m:	ľE
+assignvariableop_29_adam_conv2d_23_kernel_v:@7
)assignvariableop_30_adam_conv2d_23_bias_v:@E
+assignvariableop_31_adam_conv2d_24_kernel_v:@@7
)assignvariableop_32_adam_conv2d_24_bias_v:@E
+assignvariableop_33_adam_conv2d_25_kernel_v:@@7
)assignvariableop_34_adam_conv2d_25_bias_v:@=
)assignvariableop_35_adam_dense_6_kernel_v:
ô6
'assignvariableop_36_adam_dense_6_bias_v:	ô=
)assignvariableop_37_adam_dense_7_kernel_v:
ôľ6
'assignvariableop_38_adam_dense_7_bias_v:	ľ
identity_40˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_10˘AssignVariableOp_11˘AssignVariableOp_12˘AssignVariableOp_13˘AssignVariableOp_14˘AssignVariableOp_15˘AssignVariableOp_16˘AssignVariableOp_17˘AssignVariableOp_18˘AssignVariableOp_19˘AssignVariableOp_2˘AssignVariableOp_20˘AssignVariableOp_21˘AssignVariableOp_22˘AssignVariableOp_23˘AssignVariableOp_24˘AssignVariableOp_25˘AssignVariableOp_26˘AssignVariableOp_27˘AssignVariableOp_28˘AssignVariableOp_29˘AssignVariableOp_3˘AssignVariableOp_30˘AssignVariableOp_31˘AssignVariableOp_32˘AssignVariableOp_33˘AssignVariableOp_34˘AssignVariableOp_35˘AssignVariableOp_36˘AssignVariableOp_37˘AssignVariableOp_38˘AssignVariableOp_4˘AssignVariableOp_5˘AssignVariableOp_6˘AssignVariableOp_7˘AssignVariableOp_8˘AssignVariableOp_9ě
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*
valueB(B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHŔ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B é
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ś
_output_shapesŁ
 ::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_23_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_23_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_24_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_24_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_25_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_25_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_6_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_6_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_7_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_7_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_conv2d_23_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_conv2d_23_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_conv2d_24_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_conv2d_24_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_conv2d_25_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_conv2d_25_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_dense_6_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_dense_6_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_dense_7_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp'assignvariableop_28_adam_dense_7_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv2d_23_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2d_23_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_24_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_24_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_25_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_25_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_dense_6_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_dense_6_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp)assignvariableop_37_adam_dense_7_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp'assignvariableop_38_adam_dense_7_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Š
Identity_39Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_40IdentityIdentity_39:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_40Identity_40:output:0*c
_input_shapesR
P: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
¸
L
0__inference_max_pooling2d_16_layer_call_fn_55996

inputs
identityŮ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_55416
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:r n
J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ż
E
)__inference_flatten_4_layer_call_fn_56026

inputs
identity°
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_55485a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:˙˙˙˙˙˙˙˙˙@:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
Ş

ö
B__inference_dense_7_layer_call_and_return_conditional_losses_56072

inputs2
matmul_readvariableop_resource:
ôľ.
biasadd_readvariableop_resource:	ľ
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ôľ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ľ*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľW
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľa
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙ô: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô
 
_user_specified_nameinputs
Ę


-__inference_sequential_12_layer_call_fn_55803

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:
ô
	unknown_6:	ô
	unknown_7:
ôľ
	unknown_8:	ľ
identity˘StatefulPartitionedCallĆ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_12_layer_call_and_return_conditional_losses_55522p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:˙˙˙˙˙˙˙˙˙: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

ý
D__inference_conv2d_24_layer_call_and_return_conditional_losses_55991

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity˘BiasAdd/ReadVariableOp˘Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_55416

inputs
identity˘
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:r n
J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_55971

inputs
identity˘
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:r n
J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ĺ

'__inference_dense_6_layer_call_fn_56041

inputs
unknown:
ô
	unknown_0:	ô
identity˘StatefulPartitionedCallŘ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_55498p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
éQ
ű
__inference__traced_save_56212
file_prefix/
+savev2_conv2d_23_kernel_read_readvariableop-
)savev2_conv2d_23_bias_read_readvariableop/
+savev2_conv2d_24_kernel_read_readvariableop-
)savev2_conv2d_24_bias_read_readvariableop/
+savev2_conv2d_25_kernel_read_readvariableop-
)savev2_conv2d_25_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv2d_23_kernel_m_read_readvariableop4
0savev2_adam_conv2d_23_bias_m_read_readvariableop6
2savev2_adam_conv2d_24_kernel_m_read_readvariableop4
0savev2_adam_conv2d_24_bias_m_read_readvariableop6
2savev2_adam_conv2d_25_kernel_m_read_readvariableop4
0savev2_adam_conv2d_25_bias_m_read_readvariableop4
0savev2_adam_dense_6_kernel_m_read_readvariableop2
.savev2_adam_dense_6_bias_m_read_readvariableop4
0savev2_adam_dense_7_kernel_m_read_readvariableop2
.savev2_adam_dense_7_bias_m_read_readvariableop6
2savev2_adam_conv2d_23_kernel_v_read_readvariableop4
0savev2_adam_conv2d_23_bias_v_read_readvariableop6
2savev2_adam_conv2d_24_kernel_v_read_readvariableop4
0savev2_adam_conv2d_24_bias_v_read_readvariableop6
2savev2_adam_conv2d_25_kernel_v_read_readvariableop4
0savev2_adam_conv2d_25_bias_v_read_readvariableop4
0savev2_adam_dense_6_kernel_v_read_readvariableop2
.savev2_adam_dense_6_bias_v_read_readvariableop4
0savev2_adam_dense_7_kernel_v_read_readvariableop2
.savev2_adam_dense_7_bias_v_read_readvariableop
savev2_const

identity_1˘MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: é
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*
valueB(B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH˝
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ë
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_23_kernel_read_readvariableop)savev2_conv2d_23_bias_read_readvariableop+savev2_conv2d_24_kernel_read_readvariableop)savev2_conv2d_24_bias_read_readvariableop+savev2_conv2d_25_kernel_read_readvariableop)savev2_conv2d_25_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv2d_23_kernel_m_read_readvariableop0savev2_adam_conv2d_23_bias_m_read_readvariableop2savev2_adam_conv2d_24_kernel_m_read_readvariableop0savev2_adam_conv2d_24_bias_m_read_readvariableop2savev2_adam_conv2d_25_kernel_m_read_readvariableop0savev2_adam_conv2d_25_bias_m_read_readvariableop0savev2_adam_dense_6_kernel_m_read_readvariableop.savev2_adam_dense_6_bias_m_read_readvariableop0savev2_adam_dense_7_kernel_m_read_readvariableop.savev2_adam_dense_7_bias_m_read_readvariableop2savev2_adam_conv2d_23_kernel_v_read_readvariableop0savev2_adam_conv2d_23_bias_v_read_readvariableop2savev2_adam_conv2d_24_kernel_v_read_readvariableop0savev2_adam_conv2d_24_bias_v_read_readvariableop2savev2_adam_conv2d_25_kernel_v_read_readvariableop0savev2_adam_conv2d_25_bias_v_read_readvariableop0savev2_adam_dense_6_kernel_v_read_readvariableop.savev2_adam_dense_6_bias_v_read_readvariableop0savev2_adam_dense_7_kernel_v_read_readvariableop.savev2_adam_dense_7_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *6
dtypes,
*2(	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*ő
_input_shapesă
ŕ: :@:@:@@:@:@@:@:
ô:ô:
ôľ:ľ: : : : : : : : : :@:@:@@:@:@@:@:
ô:ô:
ôľ:ľ:@:@:@@:@:@@:@:
ô:ô:
ôľ:ľ: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:&"
 
_output_shapes
:
ô:!

_output_shapes	
:ô:&	"
 
_output_shapes
:
ôľ:!


_output_shapes	
:ľ:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:&"
 
_output_shapes
:
ô:!

_output_shapes	
:ô:&"
 
_output_shapes
:
ôľ:!

_output_shapes	
:ľ:,(
&
_output_shapes
:@: 

_output_shapes
:@:, (
&
_output_shapes
:@@: !

_output_shapes
:@:,"(
&
_output_shapes
:@@: #

_output_shapes
:@:&$"
 
_output_shapes
:
ô:!%

_output_shapes	
:ô:&&"
 
_output_shapes
:
ôľ:!'

_output_shapes	
:ľ:(

_output_shapes
: 
Đ


-__inference_sequential_12_layer_call_fn_55708
input_11!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:
ô
	unknown_6:	ô
	unknown_7:
ôľ
	unknown_8:	ľ
identity˘StatefulPartitionedCallČ
StatefulPartitionedCallStatefulPartitionedCallinput_11unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_12_layer_call_and_return_conditional_losses_55660p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:˙˙˙˙˙˙˙˙˙: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
input_11
ę

)__inference_conv2d_24_layer_call_fn_55980

inputs!
unknown:@@
	unknown_0:@
identity˘StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_55455w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs

ý
D__inference_conv2d_25_layer_call_and_return_conditional_losses_56021

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity˘BiasAdd/ReadVariableOp˘Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
Ş

ö
B__inference_dense_7_layer_call_and_return_conditional_losses_55515

inputs2
matmul_readvariableop_resource:
ôľ.
biasadd_readvariableop_resource:	ľ
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ôľ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ľ*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľW
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľa
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙ô: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô
 
_user_specified_nameinputs
š@


 __inference__wrapped_model_55395
input_11P
6sequential_12_conv2d_23_conv2d_readvariableop_resource:@E
7sequential_12_conv2d_23_biasadd_readvariableop_resource:@P
6sequential_12_conv2d_24_conv2d_readvariableop_resource:@@E
7sequential_12_conv2d_24_biasadd_readvariableop_resource:@P
6sequential_12_conv2d_25_conv2d_readvariableop_resource:@@E
7sequential_12_conv2d_25_biasadd_readvariableop_resource:@H
4sequential_12_dense_6_matmul_readvariableop_resource:
ôD
5sequential_12_dense_6_biasadd_readvariableop_resource:	ôH
4sequential_12_dense_7_matmul_readvariableop_resource:
ôľD
5sequential_12_dense_7_biasadd_readvariableop_resource:	ľ
identity˘.sequential_12/conv2d_23/BiasAdd/ReadVariableOp˘-sequential_12/conv2d_23/Conv2D/ReadVariableOp˘.sequential_12/conv2d_24/BiasAdd/ReadVariableOp˘-sequential_12/conv2d_24/Conv2D/ReadVariableOp˘.sequential_12/conv2d_25/BiasAdd/ReadVariableOp˘-sequential_12/conv2d_25/Conv2D/ReadVariableOp˘,sequential_12/dense_6/BiasAdd/ReadVariableOp˘+sequential_12/dense_6/MatMul/ReadVariableOp˘,sequential_12/dense_7/BiasAdd/ReadVariableOp˘+sequential_12/dense_7/MatMul/ReadVariableOpŹ
-sequential_12/conv2d_23/Conv2D/ReadVariableOpReadVariableOp6sequential_12_conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ë
sequential_12/conv2d_23/Conv2DConv2Dinput_115sequential_12/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
paddingSAME*
strides
˘
.sequential_12/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp7sequential_12_conv2d_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ĺ
sequential_12/conv2d_23/BiasAddBiasAdd'sequential_12/conv2d_23/Conv2D:output:06sequential_12/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
sequential_12/conv2d_23/ReluRelu(sequential_12/conv2d_23/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@Ę
&sequential_12/max_pooling2d_15/MaxPoolMaxPool*sequential_12/conv2d_23/Relu:activations:0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
ksize
*
paddingVALID*
strides
Ź
-sequential_12/conv2d_24/Conv2D/ReadVariableOpReadVariableOp6sequential_12_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0ň
sequential_12/conv2d_24/Conv2DConv2D/sequential_12/max_pooling2d_15/MaxPool:output:05sequential_12/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
paddingSAME*
strides
˘
.sequential_12/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp7sequential_12_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ĺ
sequential_12/conv2d_24/BiasAddBiasAdd'sequential_12/conv2d_24/Conv2D:output:06sequential_12/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
sequential_12/conv2d_24/ReluRelu(sequential_12/conv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@Ę
&sequential_12/max_pooling2d_16/MaxPoolMaxPool*sequential_12/conv2d_24/Relu:activations:0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
ksize
*
paddingVALID*
strides
Ź
-sequential_12/conv2d_25/Conv2D/ReadVariableOpReadVariableOp6sequential_12_conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0ň
sequential_12/conv2d_25/Conv2DConv2D/sequential_12/max_pooling2d_16/MaxPool:output:05sequential_12/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
paddingSAME*
strides
˘
.sequential_12/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp7sequential_12_conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ĺ
sequential_12/conv2d_25/BiasAddBiasAdd'sequential_12/conv2d_25/Conv2D:output:06sequential_12/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
sequential_12/conv2d_25/ReluRelu(sequential_12/conv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@n
sequential_12/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙ 	  ą
sequential_12/flatten_4/ReshapeReshape*sequential_12/conv2d_25/Relu:activations:0&sequential_12/flatten_4/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
+sequential_12/dense_6/MatMul/ReadVariableOpReadVariableOp4sequential_12_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
ô*
dtype0¸
sequential_12/dense_6/MatMulMatMul(sequential_12/flatten_4/Reshape:output:03sequential_12/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô
,sequential_12/dense_6/BiasAdd/ReadVariableOpReadVariableOp5sequential_12_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:ô*
dtype0š
sequential_12/dense_6/BiasAddBiasAdd&sequential_12/dense_6/MatMul:product:04sequential_12/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô}
sequential_12/dense_6/ReluRelu&sequential_12/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô˘
+sequential_12/dense_7/MatMul/ReadVariableOpReadVariableOp4sequential_12_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
ôľ*
dtype0¸
sequential_12/dense_7/MatMulMatMul(sequential_12/dense_6/Relu:activations:03sequential_12/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľ
,sequential_12/dense_7/BiasAdd/ReadVariableOpReadVariableOp5sequential_12_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:ľ*
dtype0š
sequential_12/dense_7/BiasAddBiasAdd&sequential_12/dense_7/MatMul:product:04sequential_12/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľ
sequential_12/dense_7/SoftmaxSoftmax&sequential_12/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľw
IdentityIdentity'sequential_12/dense_7/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľŁ
NoOpNoOp/^sequential_12/conv2d_23/BiasAdd/ReadVariableOp.^sequential_12/conv2d_23/Conv2D/ReadVariableOp/^sequential_12/conv2d_24/BiasAdd/ReadVariableOp.^sequential_12/conv2d_24/Conv2D/ReadVariableOp/^sequential_12/conv2d_25/BiasAdd/ReadVariableOp.^sequential_12/conv2d_25/Conv2D/ReadVariableOp-^sequential_12/dense_6/BiasAdd/ReadVariableOp,^sequential_12/dense_6/MatMul/ReadVariableOp-^sequential_12/dense_7/BiasAdd/ReadVariableOp,^sequential_12/dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:˙˙˙˙˙˙˙˙˙: : : : : : : : : : 2`
.sequential_12/conv2d_23/BiasAdd/ReadVariableOp.sequential_12/conv2d_23/BiasAdd/ReadVariableOp2^
-sequential_12/conv2d_23/Conv2D/ReadVariableOp-sequential_12/conv2d_23/Conv2D/ReadVariableOp2`
.sequential_12/conv2d_24/BiasAdd/ReadVariableOp.sequential_12/conv2d_24/BiasAdd/ReadVariableOp2^
-sequential_12/conv2d_24/Conv2D/ReadVariableOp-sequential_12/conv2d_24/Conv2D/ReadVariableOp2`
.sequential_12/conv2d_25/BiasAdd/ReadVariableOp.sequential_12/conv2d_25/BiasAdd/ReadVariableOp2^
-sequential_12/conv2d_25/Conv2D/ReadVariableOp-sequential_12/conv2d_25/Conv2D/ReadVariableOp2\
,sequential_12/dense_6/BiasAdd/ReadVariableOp,sequential_12/dense_6/BiasAdd/ReadVariableOp2Z
+sequential_12/dense_6/MatMul/ReadVariableOp+sequential_12/dense_6/MatMul/ReadVariableOp2\
,sequential_12/dense_7/BiasAdd/ReadVariableOp,sequential_12/dense_7/BiasAdd/ReadVariableOp2Z
+sequential_12/dense_7/MatMul/ReadVariableOp+sequential_12/dense_7/MatMul/ReadVariableOp:Y U
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
input_11

g
K__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_55404

inputs
identity˘
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:r n
J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ę


-__inference_sequential_12_layer_call_fn_55828

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:
ô
	unknown_6:	ô
	unknown_7:
ôľ
	unknown_8:	ľ
identity˘StatefulPartitionedCallĆ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_12_layer_call_and_return_conditional_losses_55660p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:˙˙˙˙˙˙˙˙˙: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ľ

ö
B__inference_dense_6_layer_call_and_return_conditional_losses_55498

inputs2
matmul_readvariableop_resource:
ô.
biasadd_readvariableop_resource:	ô
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ô*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ôs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ô*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ôQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ôb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ôw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ň$
÷
H__inference_sequential_12_layer_call_and_return_conditional_losses_55522

inputs)
conv2d_23_55438:@
conv2d_23_55440:@)
conv2d_24_55456:@@
conv2d_24_55458:@)
conv2d_25_55474:@@
conv2d_25_55476:@!
dense_6_55499:
ô
dense_6_55501:	ô!
dense_7_55516:
ôľ
dense_7_55518:	ľ
identity˘!conv2d_23/StatefulPartitionedCall˘!conv2d_24/StatefulPartitionedCall˘!conv2d_25/StatefulPartitionedCall˘dense_6/StatefulPartitionedCall˘dense_7/StatefulPartitionedCallů
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_23_55438conv2d_23_55440*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_23_layer_call_and_return_conditional_losses_55437ó
 max_pooling2d_15/PartitionedCallPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_55404
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_15/PartitionedCall:output:0conv2d_24_55456conv2d_24_55458*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_55455ó
 max_pooling2d_16/PartitionedCallPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_55416
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_16/PartitionedCall:output:0conv2d_25_55474conv2d_25_55476*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_25_layer_call_and_return_conditional_losses_55473Ţ
flatten_4/PartitionedCallPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_55485
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_6_55499dense_6_55501*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_55498
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_55516dense_7_55518*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_55515x
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľö
NoOpNoOp"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:˙˙˙˙˙˙˙˙˙: : : : : : : : : : 2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Đ


-__inference_sequential_12_layer_call_fn_55545
input_11!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:
ô
	unknown_6:	ô
	unknown_7:
ôľ
	unknown_8:	ľ
identity˘StatefulPartitionedCallČ
StatefulPartitionedCallStatefulPartitionedCallinput_11unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_12_layer_call_and_return_conditional_losses_55522p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:˙˙˙˙˙˙˙˙˙: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
input_11
Ĺ

'__inference_dense_7_layer_call_fn_56061

inputs
unknown:
ôľ
	unknown_0:	ľ
identity˘StatefulPartitionedCallŘ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_55515p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙ô: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô
 
_user_specified_nameinputs
Ć
`
D__inference_flatten_4_layer_call_and_return_conditional_losses_56032

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙ 	  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:˙˙˙˙˙˙˙˙˙@:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
Ř$
ů
H__inference_sequential_12_layer_call_and_return_conditional_losses_55740
input_11)
conv2d_23_55711:@
conv2d_23_55713:@)
conv2d_24_55717:@@
conv2d_24_55719:@)
conv2d_25_55723:@@
conv2d_25_55725:@!
dense_6_55729:
ô
dense_6_55731:	ô!
dense_7_55734:
ôľ
dense_7_55736:	ľ
identity˘!conv2d_23/StatefulPartitionedCall˘!conv2d_24/StatefulPartitionedCall˘!conv2d_25/StatefulPartitionedCall˘dense_6/StatefulPartitionedCall˘dense_7/StatefulPartitionedCallű
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCallinput_11conv2d_23_55711conv2d_23_55713*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_23_layer_call_and_return_conditional_losses_55437ó
 max_pooling2d_15/PartitionedCallPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_55404
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_15/PartitionedCall:output:0conv2d_24_55717conv2d_24_55719*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_55455ó
 max_pooling2d_16/PartitionedCallPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_55416
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_16/PartitionedCall:output:0conv2d_25_55723conv2d_25_55725*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_25_layer_call_and_return_conditional_losses_55473Ţ
flatten_4/PartitionedCallPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_55485
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_6_55729dense_6_55731*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_55498
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_55734dense_7_55736*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_55515x
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľö
NoOpNoOp"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:˙˙˙˙˙˙˙˙˙: : : : : : : : : : 2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:Y U
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
input_11

ý
D__inference_conv2d_24_layer_call_and_return_conditional_losses_55455

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity˘BiasAdd/ReadVariableOp˘Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
Ň$
÷
H__inference_sequential_12_layer_call_and_return_conditional_losses_55660

inputs)
conv2d_23_55631:@
conv2d_23_55633:@)
conv2d_24_55637:@@
conv2d_24_55639:@)
conv2d_25_55643:@@
conv2d_25_55645:@!
dense_6_55649:
ô
dense_6_55651:	ô!
dense_7_55654:
ôľ
dense_7_55656:	ľ
identity˘!conv2d_23/StatefulPartitionedCall˘!conv2d_24/StatefulPartitionedCall˘!conv2d_25/StatefulPartitionedCall˘dense_6/StatefulPartitionedCall˘dense_7/StatefulPartitionedCallů
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_23_55631conv2d_23_55633*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_23_layer_call_and_return_conditional_losses_55437ó
 max_pooling2d_15/PartitionedCallPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_55404
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_15/PartitionedCall:output:0conv2d_24_55637conv2d_24_55639*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_55455ó
 max_pooling2d_16/PartitionedCallPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_55416
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_16/PartitionedCall:output:0conv2d_25_55643conv2d_25_55645*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_25_layer_call_and_return_conditional_losses_55473Ţ
flatten_4/PartitionedCallPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_55485
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_6_55649dense_6_55651*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙ô*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_55498
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_55654dense_7_55656*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_55515x
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľö
NoOpNoOp"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:˙˙˙˙˙˙˙˙˙: : : : : : : : : : 2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ę

)__inference_conv2d_25_layer_call_fn_56010

inputs!
unknown:@@
	unknown_0:@
identity˘StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_25_layer_call_and_return_conditional_losses_55473w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
ę

)__inference_conv2d_23_layer_call_fn_55950

inputs!
unknown:@
	unknown_0:@
identity˘StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_23_layer_call_and_return_conditional_losses_55437w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

ý
D__inference_conv2d_25_layer_call_and_return_conditional_losses_55473

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity˘BiasAdd/ReadVariableOp˘Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs

ý
D__inference_conv2d_23_layer_call_and_return_conditional_losses_55961

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity˘BiasAdd/ReadVariableOp˘Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

ý
D__inference_conv2d_23_layer_call_and_return_conditional_losses_55437

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity˘BiasAdd/ReadVariableOp˘Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs"ŰL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ľ
serving_defaultĄ
E
input_119
serving_default_input_11:0˙˙˙˙˙˙˙˙˙<
dense_71
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙ľtensorflow/serving/predict:ę
Đ
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
ť

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
Ľ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
ť

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses"
_tf_keras_layer
Ľ
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses"
_tf_keras_layer
ť

.kernel
/bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses"
_tf_keras_layer
Ľ
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses"
_tf_keras_layer
ť

<kernel
=bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses"
_tf_keras_layer
ť

Dkernel
Ebias
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses"
_tf_keras_layer

Liter

Mbeta_1

Nbeta_2
	Odecay
Plearning_ratemm m!m.m/m<m=mDmEmvv v!v.v/v<v=vDvEv"
	optimizer
f
0
1
 2
!3
.4
/5
<6
=7
D8
E9"
trackable_list_wrapper
f
0
1
 2
!3
.4
/5
<6
=7
D8
E9"
trackable_list_wrapper
 "
trackable_list_wrapper
Ę
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
2˙
-__inference_sequential_12_layer_call_fn_55545
-__inference_sequential_12_layer_call_fn_55803
-__inference_sequential_12_layer_call_fn_55828
-__inference_sequential_12_layer_call_fn_55708Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
î2ë
H__inference_sequential_12_layer_call_and_return_conditional_losses_55871
H__inference_sequential_12_layer_call_and_return_conditional_losses_55914
H__inference_sequential_12_layer_call_and_return_conditional_losses_55740
H__inference_sequential_12_layer_call_and_return_conditional_losses_55772Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ĚBÉ
 __inference__wrapped_model_55395input_11"
˛
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
,
Vserving_default"
signature_map
*:(@2conv2d_23/kernel
:@2conv2d_23/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ó2Đ
)__inference_conv2d_23_layer_call_fn_55950˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
î2ë
D__inference_conv2d_23_layer_call_and_return_conditional_losses_55961˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_max_pooling2d_15_layer_call_fn_55966˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ő2ň
K__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_55971˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
*:(@@2conv2d_24/kernel
:@2conv2d_24/bias
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
Ó2Đ
)__inference_conv2d_24_layer_call_fn_55980˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
î2ë
D__inference_conv2d_24_layer_call_and_return_conditional_losses_55991˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_max_pooling2d_16_layer_call_fn_55996˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ő2ň
K__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_56001˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
*:(@@2conv2d_25/kernel
:@2conv2d_25/bias
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
Ó2Đ
)__inference_conv2d_25_layer_call_fn_56010˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
î2ë
D__inference_conv2d_25_layer_call_and_return_conditional_losses_56021˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
Ó2Đ
)__inference_flatten_4_layer_call_fn_56026˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
î2ë
D__inference_flatten_4_layer_call_and_return_conditional_losses_56032˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
": 
ô2dense_6/kernel
:ô2dense_6/bias
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
Ń2Î
'__inference_dense_6_layer_call_fn_56041˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ě2é
B__inference_dense_6_layer_call_and_return_conditional_losses_56052˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
": 
ôľ2dense_7/kernel
:ľ2dense_7/bias
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
Ń2Î
'__inference_dense_7_layer_call_fn_56061˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ě2é
B__inference_dense_7_layer_call_and_return_conditional_losses_56072˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
/
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ËBČ
#__inference_signature_wrapper_55941input_11"
˛
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

total

count
	variables
	keras_api"
_tf_keras_metric
c

total

count

_fn_kwargs
	variables
	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
/:-@2Adam/conv2d_23/kernel/m
!:@2Adam/conv2d_23/bias/m
/:-@@2Adam/conv2d_24/kernel/m
!:@2Adam/conv2d_24/bias/m
/:-@@2Adam/conv2d_25/kernel/m
!:@2Adam/conv2d_25/bias/m
':%
ô2Adam/dense_6/kernel/m
 :ô2Adam/dense_6/bias/m
':%
ôľ2Adam/dense_7/kernel/m
 :ľ2Adam/dense_7/bias/m
/:-@2Adam/conv2d_23/kernel/v
!:@2Adam/conv2d_23/bias/v
/:-@@2Adam/conv2d_24/kernel/v
!:@2Adam/conv2d_24/bias/v
/:-@@2Adam/conv2d_25/kernel/v
!:@2Adam/conv2d_25/bias/v
':%
ô2Adam/dense_6/kernel/v
 :ô2Adam/dense_6/bias/v
':%
ôľ2Adam/dense_7/kernel/v
 :ľ2Adam/dense_7/bias/v
 __inference__wrapped_model_55395{
 !./<=DE9˘6
/˘,
*'
input_11˙˙˙˙˙˙˙˙˙
Ş "2Ş/
-
dense_7"
dense_7˙˙˙˙˙˙˙˙˙ľ´
D__inference_conv2d_23_layer_call_and_return_conditional_losses_55961l7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙
Ş "-˘*
# 
0˙˙˙˙˙˙˙˙˙@
 
)__inference_conv2d_23_layer_call_fn_55950_7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙
Ş " ˙˙˙˙˙˙˙˙˙@´
D__inference_conv2d_24_layer_call_and_return_conditional_losses_55991l !7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙@
Ş "-˘*
# 
0˙˙˙˙˙˙˙˙˙@
 
)__inference_conv2d_24_layer_call_fn_55980_ !7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙@
Ş " ˙˙˙˙˙˙˙˙˙@´
D__inference_conv2d_25_layer_call_and_return_conditional_losses_56021l./7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙@
Ş "-˘*
# 
0˙˙˙˙˙˙˙˙˙@
 
)__inference_conv2d_25_layer_call_fn_56010_./7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙@
Ş " ˙˙˙˙˙˙˙˙˙@¤
B__inference_dense_6_layer_call_and_return_conditional_losses_56052^<=0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "&˘#

0˙˙˙˙˙˙˙˙˙ô
 |
'__inference_dense_6_layer_call_fn_56041Q<=0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙ô¤
B__inference_dense_7_layer_call_and_return_conditional_losses_56072^DE0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙ô
Ş "&˘#

0˙˙˙˙˙˙˙˙˙ľ
 |
'__inference_dense_7_layer_call_fn_56061QDE0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙ô
Ş "˙˙˙˙˙˙˙˙˙ľŠ
D__inference_flatten_4_layer_call_and_return_conditional_losses_56032a7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙@
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 
)__inference_flatten_4_layer_call_fn_56026T7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙@
Ş "˙˙˙˙˙˙˙˙˙î
K__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_55971R˘O
H˘E
C@
inputs4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "H˘E
>;
04˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 Ć
0__inference_max_pooling2d_15_layer_call_fn_55966R˘O
H˘E
C@
inputs4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş ";84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙î
K__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_56001R˘O
H˘E
C@
inputs4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "H˘E
>;
04˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 Ć
0__inference_max_pooling2d_16_layer_call_fn_55996R˘O
H˘E
C@
inputs4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş ";84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ă
H__inference_sequential_12_layer_call_and_return_conditional_losses_55740w
 !./<=DEA˘>
7˘4
*'
input_11˙˙˙˙˙˙˙˙˙
p 

 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙ľ
 Ă
H__inference_sequential_12_layer_call_and_return_conditional_losses_55772w
 !./<=DEA˘>
7˘4
*'
input_11˙˙˙˙˙˙˙˙˙
p

 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙ľ
 Á
H__inference_sequential_12_layer_call_and_return_conditional_losses_55871u
 !./<=DE?˘<
5˘2
(%
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙ľ
 Á
H__inference_sequential_12_layer_call_and_return_conditional_losses_55914u
 !./<=DE?˘<
5˘2
(%
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙ľ
 
-__inference_sequential_12_layer_call_fn_55545j
 !./<=DEA˘>
7˘4
*'
input_11˙˙˙˙˙˙˙˙˙
p 

 
Ş "˙˙˙˙˙˙˙˙˙ľ
-__inference_sequential_12_layer_call_fn_55708j
 !./<=DEA˘>
7˘4
*'
input_11˙˙˙˙˙˙˙˙˙
p

 
Ş "˙˙˙˙˙˙˙˙˙ľ
-__inference_sequential_12_layer_call_fn_55803h
 !./<=DE?˘<
5˘2
(%
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş "˙˙˙˙˙˙˙˙˙ľ
-__inference_sequential_12_layer_call_fn_55828h
 !./<=DE?˘<
5˘2
(%
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş "˙˙˙˙˙˙˙˙˙ľŻ
#__inference_signature_wrapper_55941
 !./<=DEE˘B
˘ 
;Ş8
6
input_11*'
input_11˙˙˙˙˙˙˙˙˙"2Ş/
-
dense_7"
dense_7˙˙˙˙˙˙˙˙˙ľ