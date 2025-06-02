function Af(e,t){for(var n=0;n<t.length;n++){const r=t[n];if(typeof r!="string"&&!Array.isArray(r)){for(const s in r)if(s!=="default"&&!(s in e)){const a=Object.getOwnPropertyDescriptor(r,s);a&&Object.defineProperty(e,s,a.get?a:{enumerable:!0,get:()=>r[s]})}}}return Object.freeze(Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}))}(function(){const t=document.createElement("link").relList;if(t&&t.supports&&t.supports("modulepreload"))return;for(const s of document.querySelectorAll('link[rel="modulepreload"]'))r(s);new MutationObserver(s=>{for(const a of s)if(a.type==="childList")for(const o of a.addedNodes)o.tagName==="LINK"&&o.rel==="modulepreload"&&r(o)}).observe(document,{childList:!0,subtree:!0});function n(s){const a={};return s.integrity&&(a.integrity=s.integrity),s.referrerPolicy&&(a.referrerPolicy=s.referrerPolicy),s.crossOrigin==="use-credentials"?a.credentials="include":s.crossOrigin==="anonymous"?a.credentials="omit":a.credentials="same-origin",a}function r(s){if(s.ep)return;s.ep=!0;const a=n(s);fetch(s.href,a)}})();/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Of=1e-7,Df=1e-4;class Ff{constructor(t,n){this.backend=t,this.dataMover=n,this.data=new WeakMap,this.dataIdsCount=0}get(t){return this.data.has(t)||this.dataMover.moveData(this.backend,t),this.data.get(t)}set(t,n){this.dataIdsCount++,this.data.set(t,n)}has(t){return this.data.has(t)}delete(t){return this.dataIdsCount--,this.data.delete(t)}numDataIds(){return this.dataIdsCount}}class Uo{refCount(t){return Nt("refCount")}incRef(t){return Nt("incRef")}timerAvailable(){return!0}time(t){return Nt("time")}read(t){return Nt("read")}readSync(t){return Nt("readSync")}readToGPU(t,n){return Nt("readToGPU")}numDataIds(){return Nt("numDataIds")}disposeData(t,n){return Nt("disposeData")}write(t,n,r){return Nt("write")}move(t,n,r,s,a){return Nt("move")}createTensorFromGPUData(t,n,r){return Nt("createTensorFromGPUData")}memory(){return Nt("memory")}floatPrecision(){return Nt("floatPrecision")}epsilon(){return this.floatPrecision()===32?Of:Df}dispose(){return Nt("dispose")}}function Nt(e){throw new Error(`'${e}' not yet implemented or not found in the registry. This kernel may not be supported by the tfjs backend you have chosen`)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qo(e){let t=e.length,n=0;for(;t>0;)n=Math.random()*t|0,t--,lr(e,t,n)}function Rf(e,t){if(e.length!==t.length)throw new Error(`Array sizes must match to be shuffled together First array length was ${e.length}Second array length was ${t.length}`);let n=e.length,r=0;for(;n>0;)r=Math.random()*n|0,n--,lr(e,n,r),lr(t,n,r)}function hn(e,t,n){return Math.max(e,Math.min(t,n))}function Bf(e){return e%2===0?e:e+1}function lr(e,t,n){const r=e[t];e[t]=e[n],e[n]=r}function Pf(e){let t=0;for(let n=0;n<e.length;n++)t+=e[n];return t}function Cf(e,t){const n=Math.random();return t*n+(1-n)*e}function Lf(e,t){let n=0;for(let r=0;r<e.length;r++){const s=Number(e[r])-Number(t[r]);n+=s*s}return n}function y(e,t){if(!e)throw new Error(typeof t=="string"?t:t())}function gt(e,t,n=""){y(Pt(e,t),()=>n+` Shapes ${e} and ${t} must match`)}function Be(e){y(e!=null,()=>"The input to the tensor constructor must be a non-null value.")}function K(e){if(e.length===0)return 1;let t=e[0];for(let n=1;n<e.length;n++)t*=e[n];return t}function zf(e){return e.length===0}function Ho(e,t){if(e===t)return!0;if(e==null||t==null||e.length!==t.length)return!1;for(let n=0;n<e.length;n++)if(e[n]!==null&&t[n]!==null&&e[n]!==t[n])return!1;return!0}function Pt(e,t){if(e===t)return!0;if(e==null||t==null||e.length!==t.length)return!1;for(let n=0;n<e.length;n++)if(e[n]!==t[n])return!1;return!0}function qe(e){return e%1===0}function Mf(e){if(Math.tanh!=null)return Math.tanh(e);if(e===1/0)return 1;if(e===-1/0)return-1;{const t=Math.exp(2*e);return(t-1)/(t+1)}}function Vf(e){const t=Math.ceil(Math.sqrt(e));return[t,Math.ceil(e/t)]}function Wf(e){const t=new Uint32Array(e);for(let n=0;n<e;++n)t[n]=n;return qo(t),t}function ln(e,t){return t<=e.length?e:e+" ".repeat(t-e.length)}function Uf(e,t=s=>0,n,r){return new Promise((s,a)=>{let o=0;const i=()=>{if(e()){s();return}o++;const u=t(o);if(n!=null&&o>=n){a();return}r!=null?r(i,u):setTimeout(i,u)};i()})}function qf(e,t){let n=1,r=-1;for(let a=0;a<e.length;++a)if(e[a]>=0)n*=e[a];else if(e[a]===-1){if(r!==-1)throw Error(`Shapes can only have 1 implicit size. Found -1 at dim ${r} and dim ${a}`);r=a}else if(e[a]<0)throw Error(`Shapes can not be < 0. Found ${e[a]} at dim ${a}`);if(r===-1){if(t>0&&t!==n)throw Error(`Size(${t}) must match the product of shape ${e}`);return e}if(n===0)throw Error(`Cannot infer the missing size in [${e}] when there are 0 elements`);if(t%n!==0)throw Error(`The implicit shape can't be a fractional number. Got ${t} / ${n}`);const s=e.slice();return s[r]=t/n,s}function _n(e,t){const n=t.length;return e=e==null?t.map((r,s)=>s):[].concat(e),y(e.every(r=>r>=-n&&r<n),()=>`All values in axis param must be in range [-${n}, ${n}) but got axis ${e}`),y(e.every(r=>qe(r)),()=>`All values in axis param must be integers but got axis ${e}`),e.map(r=>r<0?n+r:r)}function jo(e,t){const n=[],r=[],s=t!=null&&Array.isArray(t)&&t.length===0,a=t==null||s?null:_n(t,e).sort();let o=0;for(let i=0;i<e.length;++i){if(a!=null){if(a[o]===i&&e[i]!==1)throw new Error(`Can't squeeze axis ${i} since its dim '${e[i]}' is not 1`);(a[o]==null||a[o]>i)&&e[i]===1&&(n.push(e[i]),r.push(i)),a[o]<=i&&o++}e[i]!==1&&(n.push(e[i]),r.push(i))}return{newShape:n,keptDims:r}}function Go(e,t){return Os(e,t)}function Os(e,t){let n=null;if(e==null||e==="float32")n=new Float32Array(t);else if(e==="int32")n=new Int32Array(t);else if(e==="bool")n=new Uint8Array(t);else if(e==="string")n=new Array(t);else throw new Error(`Unknown data type ${e}`);return n}function Ko(e,t){for(let n=0;n<e.length;n++){const r=e[n];if(isNaN(r)||!isFinite(r))throw Error(`A tensor of type ${t} being uploaded contains ${r}.`)}}function Xo(e){return e==="bool"||e==="complex64"||e==="float32"||e==="int32"||e==="string"}function Hf(e,t){return!(t==="complex64"||t==="float32"&&e!=="complex64"||t==="int32"&&e!=="float32"&&e!=="complex64"||t==="bool"&&e==="bool")}function cr(e){if(e==="float32"||e==="int32")return 4;if(e==="complex64")return 8;if(e==="bool")return 1;throw new Error(`Unknown dtype ${e}`)}function Yo(e){if(e==null)return 0;let t=0;return e.forEach(n=>t+=n.length),t}function oe(e){return typeof e=="string"||e instanceof String}function Zo(e){return typeof e=="boolean"}function Jo(e){return typeof e=="number"}function kn(e){return Array.isArray(e)?kn(e[0]):e instanceof Float32Array?"float32":e instanceof Int32Array||e instanceof Uint8Array||e instanceof Uint8ClampedArray?"int32":Jo(e)?"float32":oe(e)?"string":Zo(e)?"bool":"float32"}function ce(e){return!!(e&&e.constructor&&e.call&&e.apply)}function hr(e,t){for(let n=t;n<e;++n)if(e%n===0)return n;return e}function Ze(e){const t=e.length;if(t<2)return[];const n=new Array(t-1);n[t-2]=e[t-1];for(let r=t-3;r>=0;--r)n[r]=n[r+1]*e[r+1];return n}function Qo(e,t,n,r=!1){const s=new Array;if(t.length===1){const a=t[0]*(r?2:1);for(let o=0;o<a;o++)s[o]=n[e+o]}else{const a=t[0],o=t.slice(1),i=o.reduce((u,l)=>u*l)*(r?2:1);for(let u=0;u<a;u++)s[u]=Qo(e+u*i,o,n,r)}return s}function _e(e,t,n=!1){if(e.length===0)return t[0];const r=e.reduce((s,a)=>s*a)*(n?2:1);if(r===0)return[];if(r!==t.length)throw new Error(`[${e}] does not match the input size ${t.length}${n?" for a complex tensor":""}.`);return Qo(0,e,t,n)}function jf(e,t){if(Array.isArray(e))return e;if(t==="float32")return e instanceof Float32Array?e:new Float32Array(e);if(t==="int32")return e instanceof Int32Array?e:new Int32Array(e);if(t==="bool"||t==="string")return Uint8Array.from(new Int32Array(e));throw new Error(`Unknown dtype ${t}`)}function Ds(e,t){const n=vr(e,t);for(let r=0;r<n.length;r++)n[r]=1;return n}function vr(e,t){if(t==null||t==="float32"||t==="complex64")return new Float32Array(e);if(t==="int32")return new Int32Array(e);if(t==="bool")return new Uint8Array(e);throw new Error(`Unknown data type ${t}`)}function Gf(e,t){const n=e.reduce((r,s)=>r*s,1);if(t==null||t==="float32")return _e(e,new Float32Array(n));if(t==="int32")return _e(e,new Int32Array(n));if(t==="bool")return _e(e,new Uint8Array(n));throw new Error(`Unknown data type ${t}`)}function $t(e){e.forEach(t=>{y(Number.isInteger(t)&&t>=0,()=>`Tensor must have a shape comprised of positive integers but got shape [${e}].`)})}function Kf(e,t,n){if(t===0)return 0;if(t===1)return e[0];let r=e[e.length-1];for(let s=0;s<e.length-1;++s)r+=n[s]*e[s];return r}function Xf(e,t,n){if(t===0)return[];if(t===1)return[e];const r=new Array(t);for(let s=0;s<r.length-1;++s)r[s]=Math.floor(e/n[s]),e-=r[s]*n[s];return r[r.length-1]=e,r}function he(e){return e&&e.then&&typeof e.then=="function"}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ga="tfjsflags";class ti{constructor(t){this.global=t,this.flags={},this.flagRegistry={},this.urlFlags={},this.getQueryParams=Yf,this.populateURLFlags()}setPlatform(t,n){this.platform!=null&&(M().getBool("IS_TEST")||M().getBool("PROD")||console.warn(`Platform ${this.platformName} has already been set. Overwriting the platform with ${t}.`)),this.platformName=t,this.platform=n}registerFlag(t,n,r){if(this.flagRegistry[t]={evaluationFn:n,setHook:r},this.urlFlags[t]!=null){const s=this.urlFlags[t];M().getBool("IS_TEST")||M().getBool("PROD")||console.warn(`Setting feature override from URL ${t}: ${s}.`),this.set(t,s)}}async getAsync(t){return t in this.flags?this.flags[t]:(this.flags[t]=await this.evaluateFlag(t),this.flags[t])}get(t){if(t in this.flags)return this.flags[t];const n=this.evaluateFlag(t);if(he(n))throw new Error(`Flag ${t} cannot be synchronously evaluated. Please use getAsync() instead.`);return this.flags[t]=n,this.flags[t]}getNumber(t){return this.get(t)}getBool(t){return this.get(t)}getString(t){return this.get(t)}getFlags(){return this.flags}get features(){return this.flags}set(t,n){if(this.flagRegistry[t]==null)throw new Error(`Cannot set flag ${t} as it has not been registered.`);this.flags[t]=n,this.flagRegistry[t].setHook!=null&&this.flagRegistry[t].setHook(n)}evaluateFlag(t){if(this.flagRegistry[t]==null)throw new Error(`Cannot evaluate flag '${t}': no evaluation function found.`);return this.flagRegistry[t].evaluationFn()}setFlags(t){this.flags=Object.assign({},t)}reset(){this.flags={},this.urlFlags={},this.populateURLFlags()}populateURLFlags(){if(typeof this.global>"u"||typeof this.global.location>"u"||typeof this.global.location.search>"u")return;const t=this.getQueryParams(this.global.location.search);Ga in t&&t[Ga].split(",").forEach(r=>{const[s,a]=r.split(":");this.urlFlags[s]=Jf(s,a)})}}function Yf(e){const t={};return e.replace(/[?&]([^=?&]+)(?:=([^&]*))?/g,(n,...r)=>(Zf(t,r[0],r[1]),r.join("="))),t}function Zf(e,t,n){e[decodeURIComponent(t)]=decodeURIComponent(n||"")}function Jf(e,t){const n=t.toLowerCase();return n==="true"||n==="false"?n==="true":`${+n}`===n?+n:t}function M(){return Fs}let Fs=null;function Qf(e){Fs=e}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let zr;function ei(){if(zr==null){let e;if(typeof window<"u")e=window;else if(typeof global<"u")e=global;else if(typeof process<"u")e=process;else if(typeof self<"u")e=self;else throw new Error("Could not find a global object");zr=e}return zr}function td(){const e=ei();return e._tfGlobals==null&&(e._tfGlobals=new Map),e._tfGlobals}function Rs(e,t){const n=td();if(n.has(e))return n.get(e);{const r=t();return n.set(e,r),n.get(e)}}const ni="Abs",ri="Acos",si="Acosh",Bs="Add",ai="AddN",oi="All",ii="Any",ui="ArgMax",li="ArgMin",ci="Asin",hi="Asinh",pi="Atan",fi="Atanh",di="Atan2",mi="AvgPool",ed="AvgPoolGrad",gi="AvgPool3D",nd="AvgPool3DGrad",yi="BatchMatMul",bi="BatchToSpaceND",wi="Bincount",Ni="BitwiseAnd",rd="BroadcastTo",vi="BroadcastArgs",Ps="Cast",Si="Ceil",Ti="ClipByValue",Ei="Complex",$i="ComplexAbs",_i="Concat",ki="Conv2D",Ii="Conv2DBackpropFilter",xi="Conv2DBackpropInput",Ai="Conv3D",sd="Conv3DBackpropFilterV2",Oi="Conv3DBackpropInputV2",Di="Cos",Fi="Cosh",Ri="Cumprod",Bi="Cumsum",Pi="CropAndResize",Ci="DenseBincount",Li="DepthToSpace",zi="DepthwiseConv2dNative",Mi="DepthwiseConv2dNativeBackpropFilter",Vi="DepthwiseConv2dNativeBackpropInput",Wi="Diag",Ui="Dilation2D",ad="Dilation2DBackpropInput",od="Dilation2DBackpropFilter",Cs="Draw",qi="RealDiv",Hi="Einsum",ji="Elu",id="EluGrad",Gi="Erf",Ki="Equal",Xi="Exp",Yi="ExpandDims",Zi="Expm1",Ji="FFT",Qi="Fill",tu="FlipLeftRight",eu="Floor",nu="FloorDiv",ru="FusedBatchNorm",su="GatherV2",au="GatherNd",ou="Greater",iu="GreaterEqual",Ls="Identity",uu="IFFT",lu="Imag",cu="IsFinite",hu="IsInf",pu="IsNan",fu="LeakyRelu",du="Less",mu="LessEqual",gu="LinSpace",yu="Log",bu="Log1p",wu="LogicalAnd",Nu="LogicalNot",vu="LogicalOr",ud="LogicalXor",ld="LogSoftmax",cd="LowerBound",Su="LRN",hd="LRNGrad",pd="MatrixBandPart",Tu="Max",Eu="Maximum",$u="MaxPool",fd="MaxPoolGrad",_u="MaxPool3D",dd="MaxPool3DGrad",ku="MaxPoolWithArgmax",Iu="Mean",xu="Min",Au="Minimum",Ou="MirrorPad",Du="Mod",Fu="Multinomial",Ru="Multiply",Bu="Neg",Pu="NotEqual",Cu="NonMaxSuppressionV3",Lu="NonMaxSuppressionV4",zu="NonMaxSuppressionV5",Mu="OnesLike",Vu="OneHot",Wu="Pack",Uu="PadV2",md="Pool",qu="Pow",Hu="Prelu",ju="Prod",Gu="RaggedGather",Ku="RaggedRange",Xu="RaggedTensorToTensor",Yu="Range",Zu="Real",Ju="Reciprocal",Qu="Relu",tl="Reshape",el="ResizeNearestNeighbor",gd="ResizeNearestNeighborGrad",nl="ResizeBilinear",yd="ResizeBilinearGrad",rl="Relu6",sl="Reverse",al="Round",ol="Rsqrt",il="ScatterNd",ul="TensorScatterUpdate",ll="SearchSorted",cl="Select",hl="Selu",pl="Slice",fl="Sin",dl="Sinh",ml="Sign",gl="Sigmoid",yl="Softplus",bl="Sqrt",wl="Sum",Nl="SpaceToBatchND",vl="SplitV",Sl="Softmax",Tl="SparseFillEmptyRows",El="SparseReshape",$l="SparseSegmentMean",_l="SparseSegmentSum",kl="SparseToDense",Il="SquaredDifference",bd="Square",xl="StaticRegexReplace",Al="StridedSlice",Ol="StringNGrams",Dl="StringSplit",Fl="StringToHashBucketFast",Rl="Sub",Bl="Tan",Pl="Tanh",zs="Tile",Cl="TopK",Ll="Transform",Jn="Transpose",zl="Unique",Ml="Unpack",Vl="UnsortedSegmentSum",wd="UpperBound",Wl="ZerosLike",Ul="Step",Jr="FromPixels",ql="RotateWithOffset",Qr="_FusedMatMul",ts="FusedConv2D",es="FusedDepthwiseConv2D";/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function se(...e){M().getBool("IS_TEST")||M().getBool("PROD")||console.warn(...e)}function Nd(...e){M().getBool("IS_TEST")||M().getBool("PROD")||console.log(...e)}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const He=Rs("kernelRegistry",()=>new Map),pn=Rs("gradRegistry",()=>new Map);function fn(e,t){const n=Ms(e,t);return He.get(n)}function ns(e){return pn.get(e)}function pr(e){const t=He.entries(),n=[];for(;;){const{done:r,value:s}=t.next();if(r)break;const[a,o]=s,[i]=a.split("_");i===e&&n.push(o)}return n}function Hl(e){const{kernelName:t,backendName:n}=e,r=Ms(t,n);He.has(r)&&se(`The kernel '${t}' for backend '${n}' is already registered`),He.set(r,e)}function vd(e){const{kernelName:t}=e;pn.has(t)&&M().getBool("DEBUG")&&se(`Overriding the gradient for '${t}'`),pn.set(t,e)}function Sd(e,t){const n=Ms(e,t);if(!He.has(n))throw new Error(`The kernel '${e}' for backend '${t}' is not registered`);He.delete(n)}function Td(e){if(!pn.has(e))throw new Error(`The gradient '${e}' for backend is not registered`);pn.delete(e)}function Ed(e,t){pr(e).forEach(r=>{const s=Object.assign({},r,{backendName:t});Hl(s)})}function Ms(e,t){return`${t}_${e}`}/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function jl(e){return e instanceof Float32Array||e instanceof Int32Array||e instanceof Uint8Array||e instanceof Uint8ClampedArray}function $d(e){return e&&e.__esModule&&Object.prototype.hasOwnProperty.call(e,"default")?e.default:e}function _d(e){if(Object.prototype.hasOwnProperty.call(e,"__esModule"))return e;var t=e.default;if(typeof t=="function"){var n=function r(){return this instanceof r?Reflect.construct(t,arguments,this.constructor):t.apply(this,arguments)};n.prototype=t.prototype}else n={};return Object.defineProperty(n,"__esModule",{value:!0}),Object.keys(e).forEach(function(r){var s=Object.getOwnPropertyDescriptor(e,r);Object.defineProperty(n,r,s.get?s:{enumerable:!0,get:function(){return e[r]}})}),n}var Mr,Ka;function kd(){if(Ka)return Mr;Ka=1,Mr=t;var e=null;try{e=new WebAssembly.Instance(new WebAssembly.Module(new Uint8Array([0,97,115,109,1,0,0,0,1,13,2,96,0,1,127,96,4,127,127,127,127,1,127,3,7,6,0,1,1,1,1,1,6,6,1,127,1,65,0,11,7,50,6,3,109,117,108,0,1,5,100,105,118,95,115,0,2,5,100,105,118,95,117,0,3,5,114,101,109,95,115,0,4,5,114,101,109,95,117,0,5,8,103,101,116,95,104,105,103,104,0,0,10,191,1,6,4,0,35,0,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,126,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,127,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,128,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,129,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,130,34,4,66,32,135,167,36,0,32,4,167,11])),{}).exports}catch{}function t(_,b,D){this.low=_|0,this.high=b|0,this.unsigned=!!D}t.prototype.__isLong__,Object.defineProperty(t.prototype,"__isLong__",{value:!0});function n(_){return(_&&_.__isLong__)===!0}t.isLong=n;var r={},s={};function a(_,b){var D,P,C;return b?(_>>>=0,(C=0<=_&&_<256)&&(P=s[_],P)?P:(D=i(_,(_|0)<0?-1:0,!0),C&&(s[_]=D),D)):(_|=0,(C=-128<=_&&_<128)&&(P=r[_],P)?P:(D=i(_,_<0?-1:0,!1),C&&(r[_]=D),D))}t.fromInt=a;function o(_,b){if(isNaN(_))return b?x:T;if(b){if(_<0)return x;if(_>=g)return F}else{if(_<=-9223372036854776e3)return R;if(_+1>=N)return A}return _<0?o(-_,b).neg():i(_%d|0,_/d|0,b)}t.fromNumber=o;function i(_,b,D){return new t(_,b,D)}t.fromBits=i;var u=Math.pow;function l(_,b,D){if(_.length===0)throw Error("empty string");if(_==="NaN"||_==="Infinity"||_==="+Infinity"||_==="-Infinity")return T;if(typeof b=="number"?(D=b,b=!1):b=!!b,D=D||10,D<2||36<D)throw RangeError("radix");var P;if((P=_.indexOf("-"))>0)throw Error("interior hyphen");if(P===0)return l(_.substring(1),b,D).neg();for(var C=o(u(D,8)),L=T,q=0;q<_.length;q+=8){var G=Math.min(8,_.length-q),tt=parseInt(_.substring(q,q+G),D);if(G<8){var Z=o(u(D,G));L=L.mul(Z).add(o(tt))}else L=L.mul(C),L=L.add(o(tt))}return L.unsigned=b,L}t.fromString=l;function h(_,b){return typeof _=="number"?o(_,b):typeof _=="string"?l(_,b):i(_.low,_.high,typeof b=="boolean"?b:_.unsigned)}t.fromValue=h;var c=65536,p=1<<24,d=c*c,g=d*d,N=g/2,w=a(p),T=a(0);t.ZERO=T;var x=a(0,!0);t.UZERO=x;var $=a(1);t.ONE=$;var E=a(1,!0);t.UONE=E;var I=a(-1);t.NEG_ONE=I;var A=i(-1,2147483647,!1);t.MAX_VALUE=A;var F=i(-1,-1,!0);t.MAX_UNSIGNED_VALUE=F;var R=i(0,-2147483648,!1);t.MIN_VALUE=R;var k=t.prototype;return k.toInt=function(){return this.unsigned?this.low>>>0:this.low},k.toNumber=function(){return this.unsigned?(this.high>>>0)*d+(this.low>>>0):this.high*d+(this.low>>>0)},k.toString=function(b){if(b=b||10,b<2||36<b)throw RangeError("radix");if(this.isZero())return"0";if(this.isNegative())if(this.eq(R)){var D=o(b),P=this.div(D),C=P.mul(D).sub(this);return P.toString(b)+C.toInt().toString(b)}else return"-"+this.neg().toString(b);for(var L=o(u(b,6),this.unsigned),q=this,G="";;){var tt=q.div(L),Z=q.sub(tt.mul(L)).toInt()>>>0,nt=Z.toString(b);if(q=tt,q.isZero())return nt+G;for(;nt.length<6;)nt="0"+nt;G=""+nt+G}},k.getHighBits=function(){return this.high},k.getHighBitsUnsigned=function(){return this.high>>>0},k.getLowBits=function(){return this.low},k.getLowBitsUnsigned=function(){return this.low>>>0},k.getNumBitsAbs=function(){if(this.isNegative())return this.eq(R)?64:this.neg().getNumBitsAbs();for(var b=this.high!=0?this.high:this.low,D=31;D>0&&(b&1<<D)==0;D--);return this.high!=0?D+33:D+1},k.isZero=function(){return this.high===0&&this.low===0},k.eqz=k.isZero,k.isNegative=function(){return!this.unsigned&&this.high<0},k.isPositive=function(){return this.unsigned||this.high>=0},k.isOdd=function(){return(this.low&1)===1},k.isEven=function(){return(this.low&1)===0},k.equals=function(b){return n(b)||(b=h(b)),this.unsigned!==b.unsigned&&this.high>>>31===1&&b.high>>>31===1?!1:this.high===b.high&&this.low===b.low},k.eq=k.equals,k.notEquals=function(b){return!this.eq(b)},k.neq=k.notEquals,k.ne=k.notEquals,k.lessThan=function(b){return this.comp(b)<0},k.lt=k.lessThan,k.lessThanOrEqual=function(b){return this.comp(b)<=0},k.lte=k.lessThanOrEqual,k.le=k.lessThanOrEqual,k.greaterThan=function(b){return this.comp(b)>0},k.gt=k.greaterThan,k.greaterThanOrEqual=function(b){return this.comp(b)>=0},k.gte=k.greaterThanOrEqual,k.ge=k.greaterThanOrEqual,k.compare=function(b){if(n(b)||(b=h(b)),this.eq(b))return 0;var D=this.isNegative(),P=b.isNegative();return D&&!P?-1:!D&&P?1:this.unsigned?b.high>>>0>this.high>>>0||b.high===this.high&&b.low>>>0>this.low>>>0?-1:1:this.sub(b).isNegative()?-1:1},k.comp=k.compare,k.negate=function(){return!this.unsigned&&this.eq(R)?R:this.not().add($)},k.neg=k.negate,k.add=function(b){n(b)||(b=h(b));var D=this.high>>>16,P=this.high&65535,C=this.low>>>16,L=this.low&65535,q=b.high>>>16,G=b.high&65535,tt=b.low>>>16,Z=b.low&65535,nt=0,wt=0,ot=0,yt=0;return yt+=L+Z,ot+=yt>>>16,yt&=65535,ot+=C+tt,wt+=ot>>>16,ot&=65535,wt+=P+G,nt+=wt>>>16,wt&=65535,nt+=D+q,nt&=65535,i(ot<<16|yt,nt<<16|wt,this.unsigned)},k.subtract=function(b){return n(b)||(b=h(b)),this.add(b.neg())},k.sub=k.subtract,k.multiply=function(b){if(this.isZero())return T;if(n(b)||(b=h(b)),e){var D=e.mul(this.low,this.high,b.low,b.high);return i(D,e.get_high(),this.unsigned)}if(b.isZero())return T;if(this.eq(R))return b.isOdd()?R:T;if(b.eq(R))return this.isOdd()?R:T;if(this.isNegative())return b.isNegative()?this.neg().mul(b.neg()):this.neg().mul(b).neg();if(b.isNegative())return this.mul(b.neg()).neg();if(this.lt(w)&&b.lt(w))return o(this.toNumber()*b.toNumber(),this.unsigned);var P=this.high>>>16,C=this.high&65535,L=this.low>>>16,q=this.low&65535,G=b.high>>>16,tt=b.high&65535,Z=b.low>>>16,nt=b.low&65535,wt=0,ot=0,yt=0,Mn=0;return Mn+=q*nt,yt+=Mn>>>16,Mn&=65535,yt+=L*nt,ot+=yt>>>16,yt&=65535,yt+=q*Z,ot+=yt>>>16,yt&=65535,ot+=C*nt,wt+=ot>>>16,ot&=65535,ot+=L*Z,wt+=ot>>>16,ot&=65535,ot+=q*tt,wt+=ot>>>16,ot&=65535,wt+=P*nt+C*Z+L*tt+q*G,wt&=65535,i(yt<<16|Mn,wt<<16|ot,this.unsigned)},k.mul=k.multiply,k.divide=function(b){if(n(b)||(b=h(b)),b.isZero())throw Error("division by zero");if(e){if(!this.unsigned&&this.high===-2147483648&&b.low===-1&&b.high===-1)return this;var D=(this.unsigned?e.div_u:e.div_s)(this.low,this.high,b.low,b.high);return i(D,e.get_high(),this.unsigned)}if(this.isZero())return this.unsigned?x:T;var P,C,L;if(this.unsigned){if(b.unsigned||(b=b.toUnsigned()),b.gt(this))return x;if(b.gt(this.shru(1)))return E;L=x}else{if(this.eq(R)){if(b.eq($)||b.eq(I))return R;if(b.eq(R))return $;var q=this.shr(1);return P=q.div(b).shl(1),P.eq(T)?b.isNegative()?$:I:(C=this.sub(b.mul(P)),L=P.add(C.div(b)),L)}else if(b.eq(R))return this.unsigned?x:T;if(this.isNegative())return b.isNegative()?this.neg().div(b.neg()):this.neg().div(b).neg();if(b.isNegative())return this.div(b.neg()).neg();L=T}for(C=this;C.gte(b);){P=Math.max(1,Math.floor(C.toNumber()/b.toNumber()));for(var G=Math.ceil(Math.log(P)/Math.LN2),tt=G<=48?1:u(2,G-48),Z=o(P),nt=Z.mul(b);nt.isNegative()||nt.gt(C);)P-=tt,Z=o(P,this.unsigned),nt=Z.mul(b);Z.isZero()&&(Z=$),L=L.add(Z),C=C.sub(nt)}return L},k.div=k.divide,k.modulo=function(b){if(n(b)||(b=h(b)),e){var D=(this.unsigned?e.rem_u:e.rem_s)(this.low,this.high,b.low,b.high);return i(D,e.get_high(),this.unsigned)}return this.sub(this.div(b).mul(b))},k.mod=k.modulo,k.rem=k.modulo,k.not=function(){return i(~this.low,~this.high,this.unsigned)},k.and=function(b){return n(b)||(b=h(b)),i(this.low&b.low,this.high&b.high,this.unsigned)},k.or=function(b){return n(b)||(b=h(b)),i(this.low|b.low,this.high|b.high,this.unsigned)},k.xor=function(b){return n(b)||(b=h(b)),i(this.low^b.low,this.high^b.high,this.unsigned)},k.shiftLeft=function(b){return n(b)&&(b=b.toInt()),(b&=63)===0?this:b<32?i(this.low<<b,this.high<<b|this.low>>>32-b,this.unsigned):i(0,this.low<<b-32,this.unsigned)},k.shl=k.shiftLeft,k.shiftRight=function(b){return n(b)&&(b=b.toInt()),(b&=63)===0?this:b<32?i(this.low>>>b|this.high<<32-b,this.high>>b,this.unsigned):i(this.high>>b-32,this.high>=0?0:-1,this.unsigned)},k.shr=k.shiftRight,k.shiftRightUnsigned=function(b){if(n(b)&&(b=b.toInt()),b&=63,b===0)return this;var D=this.high;if(b<32){var P=this.low;return i(P>>>b|D<<32-b,D>>>b,this.unsigned)}else return b===32?i(D,0,this.unsigned):i(D>>>b-32,0,this.unsigned)},k.shru=k.shiftRightUnsigned,k.shr_u=k.shiftRightUnsigned,k.toSigned=function(){return this.unsigned?i(this.low,this.high,!1):this},k.toUnsigned=function(){return this.unsigned?this:i(this.low,this.high,!0)},k.toBytes=function(b){return b?this.toBytesLE():this.toBytesBE()},k.toBytesLE=function(){var b=this.high,D=this.low;return[D&255,D>>>8&255,D>>>16&255,D>>>24,b&255,b>>>8&255,b>>>16&255,b>>>24]},k.toBytesBE=function(){var b=this.high,D=this.low;return[b>>>24,b>>>16&255,b>>>8&255,b&255,D>>>24,D>>>16&255,D>>>8&255,D&255]},t.fromBytes=function(b,D,P){return P?t.fromBytesLE(b,D):t.fromBytesBE(b,D)},t.fromBytesLE=function(b,D){return new t(b[0]|b[1]<<8|b[2]<<16|b[3]<<24,b[4]|b[5]<<8|b[6]<<16|b[7]<<24,D)},t.fromBytesBE=function(b,D){return new t(b[4]<<24|b[5]<<16|b[6]<<8|b[7],b[0]<<24|b[1]<<16|b[2]<<8|b[3],D)},Mr}var Gl=kd();const Kl=$d(Gl),Id=Af({__proto__:null,default:Kl},[Gl]);/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Se=Kl||Id;function In(e){return Se.fromString(e,!0,16)}const Xl=In("c3a5c85c97cb3127"),ve=In("b492b66fbe98f273"),ft=In("9ae16a3b2f90404f");function rs(e){return e.xor(e.shru(47))}function Yl(e,t,n){const r=e.slice(t,t+n);return Se.fromBytes(Array.from(r),!0,!0)}function J(e,t){return Yl(e,t,8)}function Xa(e,t){return Yl(e,t,4)}function it(e,t){return t===0?e:e.shru(t).or(e.shl(64-t))}function le(e,t,n=In("9ddfea08eb382d69")){let r=e.xor(t).mul(n);r=r.xor(r.shru(47));let s=t.xor(r).mul(n);return s=s.xor(s.shru(47)),s=s.mul(n),s}function xd(e,t,n,r,s,a){s=s.add(e),a=it(a.add(s).add(r),21);const o=s;return s=s.add(t),s=s.add(n),a=a.add(it(s,44)),[s.add(r),a.add(o)]}function Vn(e,t,n,r){return xd(J(e,t),J(e,t+8),J(e,t+16),J(e,t+24),n,r)}function Ad(e,t=e.length){if(t>=8){const n=ft.add(t*2),r=J(e,0).add(ft),s=J(e,t-8),a=it(s,37).mul(n).add(r),o=it(r,25).add(s).mul(n);return le(a,o,n)}if(t>=4){const n=ft.add(t*2),r=Xa(e,0);return le(r.shl(3).add(t),Xa(e,t-4),n)}if(t>0){const n=e[0],r=e[t>>1],s=e[t-1],a=n+(r<<8),o=t+(s<<2);return rs(ft.mul(a).xor(Xl.mul(o))).mul(ft)}return ft}function Od(e,t=e.length){const n=ft.add(t*2),r=J(e,0).mul(ve),s=J(e,8),a=J(e,t-8).mul(n),o=J(e,t-16).mul(ft);return le(it(r.add(s),43).add(it(a,30)).add(o),r.add(it(s.add(ft),18)).add(a),n)}function Dd(e,t=e.length){const n=ft.add(t*2),r=J(e,0).mul(ft),s=J(e,8),a=J(e,t-8).mul(n),o=J(e,t-16).mul(ft),i=it(r.add(s),43).add(it(a,30)).add(o),u=le(i,r.add(it(s.add(ft),18)).add(a),n),l=J(e,16).mul(n),h=J(e,24),c=i.add(J(e,t-32)).mul(n),p=u.add(J(e,t-24)).mul(n);return le(it(l.add(h),43).add(it(c,30)).add(p),l.add(it(h.add(r),18)).add(c),n)}function Fd(e,t=e.length){const n=Se.fromNumber(81,!0);if(t<=32)return t<=16?Ad(e,t):Od(e,t);if(t<=64)return Dd(e,t);let r=n,s=n.mul(ve).add(113),a=rs(s.mul(ft).add(113)).mul(ft),o=[Se.UZERO,Se.UZERO],i=[Se.UZERO,Se.UZERO];r=r.mul(ft).add(J(e,0));let u=0;const l=(t-1>>6)*64,h=l+(t-1&63)-63;do r=it(r.add(s).add(o[0]).add(J(e,u+8)),37).mul(ve),s=it(s.add(o[1]).add(J(e,u+48)),42).mul(ve),r=r.xor(i[1]),s=s.add(o[0]).add(J(e,u+40)),a=it(a.add(i[0]),33).mul(ve),o=Vn(e,u,o[1].mul(ve),r.add(i[0])),i=Vn(e,u+32,a.add(i[1]),s.add(J(e,u+16))),[a,r]=[r,a],u+=64;while(u!==l);const c=ve.add(a.and(255).shl(1));return u=h,i[0]=i[0].add(t-1&63),o[0]=o[0].add(i[0]),i[0]=i[0].add(o[0]),r=it(r.add(s).add(o[0]).add(J(e,u+8)),37).mul(c),s=it(s.add(o[1]).add(J(e,u+48)),42).mul(c),r=r.xor(i[1].mul(9)),s=s.add(o[0].mul(9).add(J(e,u+40))),a=it(a.add(i[0]),33).mul(c),o=Vn(e,u,o[1].mul(c),r.add(i[0])),i=Vn(e,u+32,a.add(i[1]),s.add(J(e,u+16))),[a,r]=[r,a],le(le(o[0],i[0],c).add(rs(s).mul(Xl)).add(a),le(o[1],i[1],c).add(r),c)}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Rd(e,t){return t==="string"?xn(e):Sr([e],t)}function Bd(e,t){return e instanceof Float32Array&&t==="float32"||e instanceof Int32Array&&t==="int32"||e instanceof Uint8Array&&t==="bool"}function Sr(e,t){if(t==="string")throw new Error("Cannot convert a string[] to a TypedArray");if(Array.isArray(e)&&(e=pe(e)),M().getBool("DEBUG")&&Ko(e,t),Bd(e,t))return e;if(t==null||t==="float32"||t==="complex64")return new Float32Array(e);if(t==="int32")return new Int32Array(e);if(t==="bool"){const n=new Uint8Array(e.length);for(let r=0;r<n.length;++r)Math.round(e[r])!==0&&(n[r]=1);return n}else throw new Error(`Unknown data type ${t}`)}function dn(){return M().platform.now()}function Pd(e,t){return M().platform.fetch(e,t)}function xn(e,t="utf-8"){return t=t||"utf-8",M().platform.encode(e,t)}function fr(e,t="utf-8"){return t=t||"utf-8",M().platform.decode(e,t)}function ut(e){return M().platform.isTypedArray!=null?M().platform.isTypedArray(e):jl(e)}function pe(e,t=[],n=!1){if(t==null&&(t=[]),typeof e=="boolean"||typeof e=="number"||typeof e=="string"||he(e)||e==null||ut(e)&&n)t.push(e);else if(Array.isArray(e)||ut(e))for(let r=0;r<e.length;++r)pe(e[r],t,n);else{let r=-1;for(const s of Object.keys(e))/^([1-9]+[0-9]*|0)$/.test(s)&&(r=Math.max(r,Number(s)));for(let s=0;s<=r;s++)pe(e[s],t,n)}return t}const Cd=Object.freeze(Object.defineProperty({__proto__:null,arraysEqual:Pt,arraysEqualWithNull:Ho,assert:y,assertNonNegativeIntegerDimensions:$t,assertNonNull:Be,assertShapesMatch:gt,bytesFromStringArray:Yo,bytesPerElement:cr,checkConversionForErrors:Ko,clamp:hn,computeStrides:Ze,convertBackendValuesAndArrayBuffer:jf,createScalarValue:Rd,createShuffledIndices:Wf,decodeString:fr,distSquared:Lf,encodeString:xn,fetch:Pd,fingerPrint64:Fd,flatten:pe,getArrayFromDType:Os,getTypedArrayFromDType:Go,hasEncodingLoss:Hf,hexToLong:In,indexToLoc:Xf,inferDtype:kn,inferFromImplicitShape:qf,isBoolean:Zo,isFunction:ce,isInt:qe,isNumber:Jo,isPromise:he,isScalarShape:zf,isString:oe,isTypedArray:ut,isValidDtype:Xo,locToIndex:Kf,makeOnesTypedArray:Ds,makeZerosNestedTypedArray:Gf,makeZerosTypedArray:vr,nearestDivisor:hr,nearestLargerEven:Bf,now:dn,parseAxisParam:_n,randUniform:Cf,repeatedTry:Uf,rightPad:ln,shuffle:qo,shuffleCombo:Rf,sizeFromShape:K,sizeToSquarishShape:Vf,squeezeShape:jo,sum:Pf,swap:lr,tanh:Mf,toNestedArray:_e,toTypedArray:Sr},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Ld{constructor(t,n){this.backendTimer=t,this.logger=n,n==null&&(this.logger=new Md)}profileKernel(t,n,r){let s;const a=()=>{s=r()};let o;const i=dn();if(this.backendTimer.timerAvailable())o=this.backendTimer.time(a);else{a();for(const l of s)l.dataSync();o=Promise.resolve({kernelMs:dn()-i})}if(M().getBool("CHECK_COMPUTATION_FOR_ERRORS"))for(let l=0;l<s.length;l++){const h=s[l];h.data().then(c=>{zd(c,h.dtype,t)})}return{kernelName:t,outputs:s,inputs:n,timeMs:o.then(l=>l.kernelMs),extraInfo:o.then(l=>l.getExtraProfileInfo!=null?l.getExtraProfileInfo():"")}}logKernelProfile(t){const{kernelName:n,outputs:r,timeMs:s,inputs:a,extraInfo:o}=t;r.forEach(i=>{Promise.all([i.data(),s,o]).then(u=>{this.logger.logKernelProfile(n,i,u[0],u[1],a,u[2])})})}}function zd(e,t,n){if(t!=="float32")return!1;for(let r=0;r<e.length;r++){const s=e[r];if(isNaN(s)||!isFinite(s))return console.warn(`Found ${s} in the result of '${n}'`),!0}return!1}class Md{logKernelProfile(t,n,r,s,a,o){const i=typeof s=="number"?ln(`${s}ms`,9):s.error,u=ln(t,25),l=n.rank,h=n.size,c=ln(n.shape.toString(),14);let p="";for(const d in a){const g=a[d];if(g!=null){const N=g.shape||n.shape,w=N.length;p+=`${d}: ${w}D ${w>0?N:""} `}}console.log(`%c${u}	%c${i}	%c${l}D ${c}	%c${h}	%c${p}	%c${o}`,"font-weight:bold","color:red","color:blue","color: orange","color: green","color: steelblue")}}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vd(e,t,n){const r={},s={};for(let u=0;u<t.length;u++)r[t[u].id]=!0;for(let u=0;u<e.length;u++){const l=e[u],h=l.inputs;for(const c in h){const p=h[c];let d=!1;for(let g=0;g<t.length;g++)if(r[p.id]){l.outputs.forEach(N=>r[N.id]=!0),d=!0,s[l.id]=!0;break}if(d)break}}const a={};a[n.id]=!0;const o={};for(let u=e.length-1;u>=0;u--){const l=e[u],h=l.inputs;for(let c=0;c<l.outputs.length;c++)if(a[l.outputs[c].id]){for(const p in h)a[h[p].id]=!0,o[l.id]=!0;break}}const i=[];for(let u=0;u<e.length;u++){const l=e[u];if(s[l.id]&&o[l.id]){const h={};for(const p in l.inputs){const d=l.inputs[p];r[d.id]&&(h[p]=d)}const c=Object.assign({},l);c.inputs=h,c.outputs=l.outputs,i.push(c)}}return i}function Wd(e,t,n,r){for(let s=t.length-1;s>=0;s--){const a=t[s],o=[];if(a.outputs.forEach(u=>{const l=e[u.id];l!=null?o.push(l):o.push(null)}),a.gradient==null)throw new Error(`Cannot compute gradient: gradient function not found for ${a.kernelName}.`);const i=a.gradient(o);for(const u in a.inputs){if(!(u in i))throw new Error(`Cannot backprop through input ${u}. Available gradients found: ${Object.keys(i)}.`);const l=n(()=>i[u]());if(l.dtype!=="float32")throw new Error(`Error in gradient for op ${a.kernelName}. The gradient of input ${u} must have 'float32' dtype, but has '${l.dtype}'`);const h=a.inputs[u];if(!Pt(l.shape,h.shape))throw new Error(`Error in gradient for op ${a.kernelName}. The gradient of input '${u}' has shape '${l.shape}', which does not match the shape of the input '${h.shape}'`);if(e[h.id]==null)e[h.id]=l;else{const c=e[h.id];e[h.id]=r(c,l),c.dispose()}}}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ya=20,nn=3,Vr=7;function Ud(e,t,n,r){const s=Ze(t),a=qd(e,t,n,s),o=t.length,i=Qn(e,t,n,s,a),u=["Tensor"];return r&&(u.push(`  dtype: ${n}`),u.push(`  rank: ${o}`),u.push(`  shape: [${t}]`),u.push("  values:")),u.push(i.map(l=>"    "+l).join(`
`)),u.join(`
`)}function qd(e,t,n,r){const s=K(t),a=r[r.length-1],o=new Array(a).fill(0),i=t.length,u=n==="complex64"?an(e):e;if(i>1)for(let l=0;l<s/a;l++){const h=l*a;for(let c=0;c<a;c++)o[c]=Math.max(o[c],sn(u[h+c],0,n).length)}return o}function sn(e,t,n){let r;return Array.isArray(e)?r=`${parseFloat(e[0].toFixed(Vr))} + ${parseFloat(e[1].toFixed(Vr))}j`:oe(e)?r=`'${e}'`:n==="bool"?r=Zl(e):r=parseFloat(e.toFixed(Vr)).toString(),ln(r,t)}function Zl(e){return e===0?"false":"true"}function Qn(e,t,n,r,s,a=!0){const o=n==="complex64"?2:1,i=t[0],u=t.length;if(u===0){if(n==="complex64"){const N=an(e);return[sn(N[0],0,n)]}return n==="bool"?[Zl(e[0])]:[e[0].toString()]}if(u===1){if(i>Ya){const w=nn*o;let T=Array.from(e.slice(0,w)),x=Array.from(e.slice((i-nn)*o,i*o));return n==="complex64"&&(T=an(T),x=an(x)),["["+T.map(($,E)=>sn($,s[E],n)).join(", ")+", ..., "+x.map(($,E)=>sn($,s[i-nn+E],n)).join(", ")+"]"]}return["["+(n==="complex64"?an(e):Array.from(e)).map((w,T)=>sn(w,s[T],n)).join(", ")+"]"]}const l=t.slice(1),h=r.slice(1),c=r[0]*o,p=[];if(i>Ya){for(let N=0;N<nn;N++){const w=N*c,T=w+c;p.push(...Qn(e.slice(w,T),l,n,h,s,!1))}p.push("...");for(let N=i-nn;N<i;N++){const w=N*c,T=w+c;p.push(...Qn(e.slice(w,T),l,n,h,s,N===i-1))}}else for(let N=0;N<i;N++){const w=N*c,T=w+c;p.push(...Qn(e.slice(w,T),l,n,h,s,N===i-1))}const d=u===2?",":"";p[0]="["+(i>0?p[0]+d:"");for(let N=1;N<p.length-1;N++)p[N]=" "+p[N]+d;let g=`,
`;for(let N=2;N<u;N++)g+=`
`;return p[p.length-1]=" "+p[p.length-1]+"]"+(a?"":g),p}function an(e){const t=[];for(let n=0;n<e.length;n+=2)t.push([e[n],e[n+1]]);return t}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class dr{constructor(t,n,r){if(this.dtype=n,this.shape=t.slice(),this.size=K(t),r!=null){const s=r.length;y(s===this.size,()=>`Length of values '${s}' does not match the size inferred by the shape '${this.size}'.`)}if(n==="complex64")throw new Error("complex64 dtype TensorBuffers are not supported. Please create a TensorBuffer for the real and imaginary parts separately and call tf.complex(real, imag).");this.values=r||Os(n,this.size),this.strides=Ze(t)}set(t,...n){n.length===0&&(n=[0]),y(n.length===this.rank,()=>`The number of provided coordinates (${n.length}) must match the rank (${this.rank})`);const r=this.locToIndex(n);this.values[r]=t}get(...t){t.length===0&&(t=[0]);let n=0;for(const s of t){if(s<0||s>=this.shape[n]){const a=`Requested out of range element at ${t}.   Buffer shape=${this.shape}`;throw new Error(a)}n++}let r=t[t.length-1];for(let s=0;s<t.length-1;++s)r+=this.strides[s]*t[s];return this.values[r]}locToIndex(t){if(this.rank===0)return 0;if(this.rank===1)return t[0];let n=t[t.length-1];for(let r=0;r<t.length-1;++r)n+=this.strides[r]*t[r];return n}indexToLoc(t){if(this.rank===0)return[];if(this.rank===1)return[t];const n=new Array(this.shape.length);for(let r=0;r<n.length-1;++r)n[r]=Math.floor(t/this.strides[r]),t-=n[r]*this.strides[r];return n[n.length-1]=t,n}get rank(){return this.shape.length}toTensor(){return Dt().makeTensor(this.values,this.shape,this.dtype)}}let Dt=null,Le=null;function Hd(e){Dt=e}function jd(e){Le=e}class et{constructor(t,n,r,s){this.kept=!1,this.isDisposedInternal=!1,this.shape=t.slice(),this.dtype=n||"float32",this.size=K(t),this.strides=Ze(t),this.dataId=r,this.id=s,this.rankType=this.rank<5?this.rank.toString():"higher"}get rank(){return this.shape.length}async buffer(){const t=await this.data();return Le.buffer(this.shape,this.dtype,t)}bufferSync(){return Le.buffer(this.shape,this.dtype,this.dataSync())}async array(){const t=await this.data();return _e(this.shape,t,this.dtype==="complex64")}arraySync(){return _e(this.shape,this.dataSync(),this.dtype==="complex64")}async data(){this.throwIfDisposed();const t=Dt().read(this.dataId);if(this.dtype==="string"){const n=await t;try{return n.map(r=>fr(r))}catch{throw new Error("Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().")}}return t}dataToGPU(t){return this.throwIfDisposed(),Dt().readToGPU(this.dataId,t)}dataSync(){this.throwIfDisposed();const t=Dt().readSync(this.dataId);if(this.dtype==="string")try{return t.map(n=>fr(n))}catch{throw new Error("Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().")}return t}async bytes(){this.throwIfDisposed();const t=await Dt().read(this.dataId);return this.dtype==="string"?t:new Uint8Array(t.buffer)}dispose(){this.isDisposed||(this.kerasMask&&this.kerasMask.dispose(),Dt().disposeTensor(this),this.isDisposedInternal=!0)}get isDisposed(){return this.isDisposedInternal}throwIfDisposed(){if(this.isDisposed)throw new Error("Tensor is disposed.")}print(t=!1){return Le.print(this,t)}clone(){return this.throwIfDisposed(),Le.clone(this)}toString(t=!1){const n=this.dataSync();return Ud(n,this.shape,this.dtype,t)}cast(t){return this.throwIfDisposed(),Le.cast(this,t)}variable(t=!0,n,r){return this.throwIfDisposed(),Dt().makeVariable(this,t,n,r)}}Object.defineProperty(et,Symbol.hasInstance,{value:e=>!!e&&e.data!=null&&e.dataSync!=null&&e.throwIfDisposed!=null});function Jl(){return Rs("Tensor",()=>et)}Jl();class mn extends et{constructor(t,n,r,s){super(t.shape,t.dtype,t.dataId,s),this.trainable=n,this.name=r}assign(t){if(t.dtype!==this.dtype)throw new Error(`dtype of the new value (${t.dtype}) and previous value (${this.dtype}) must match`);if(!Pt(t.shape,this.shape))throw new Error(`shape of the new value (${t.shape}) and previous value (${this.shape}) must match`);Dt().disposeTensor(this),this.dataId=t.dataId,Dt().incRef(this,null)}dispose(){Dt().disposeVariable(this),this.isDisposedInternal=!0}}Object.defineProperty(mn,Symbol.hasInstance,{value:e=>e instanceof et&&e.assign!=null&&e.assign instanceof Function});/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var ss;(function(e){e.R0="R0",e.R1="R1",e.R2="R2",e.R3="R3",e.R4="R4",e.R5="R5",e.R6="R6"})(ss||(ss={}));var as;(function(e){e.float32="float32",e.int32="int32",e.bool="int32",e.complex64="complex64"})(as||(as={}));var os;(function(e){e.float32="float32",e.int32="int32",e.bool="bool",e.complex64="complex64"})(os||(os={}));var is;(function(e){e.float32="float32",e.int32="float32",e.bool="float32",e.complex64="complex64"})(is||(is={}));var us;(function(e){e.float32="complex64",e.int32="complex64",e.bool="complex64",e.complex64="complex64"})(us||(us={}));const Gd={float32:is,int32:as,bool:os,complex64:us};function Tr(e,t){if(e==="string"||t==="string"){if(e==="string"&&t==="string")return"string";throw new Error(`Can not upcast ${e} with ${t}`)}return Gd[e][t]}function Kd(e){return Tr(e,"int32")}function Ql(e){return e!=null&&typeof e=="object"&&"texture"in e&&e.texture instanceof WebGLTexture}function tc(e){return typeof GPUBuffer<"u"&&e!=null&&typeof e=="object"&&"buffer"in e&&e.buffer instanceof GPUBuffer}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function st(e,t){if(e.dtype===t.dtype)return[e,t];const n=Tr(e.dtype,t.dtype);return[e.cast(n),t.cast(n)]}function ec(e,t){y(e.dtype===t.dtype,()=>`The dtypes of the first(${e.dtype}) and second(${t.dtype}) input must match`)}function Xd(e,t){return t.some(n=>n.id===e.id)}function Vs(e){const t=[];return nc(e,t,new Set),t}function nc(e,t,n){if(e==null)return;if(e instanceof et){t.push(e);return}if(!Yd(e))return;const r=e;for(const s in r){const a=r[s];n.has(a)||(n.add(a),nc(a,t,n))}}function Yd(e){return Array.isArray(e)||typeof e=="object"}const Zd=Object.freeze(Object.defineProperty({__proto__:null,assertTypesMatch:ec,getTensorsInContainer:Vs,isTensorInList:Xd,makeTypesMatch:st},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Wr(e){return e.kernelName!=null}class Za{constructor(){this.registeredVariables={},this.nextTapeNodeId=0,this.numBytes=0,this.numTensors=0,this.numStringTensors=0,this.numDataBuffers=0,this.gradientDepth=0,this.kernelDepth=0,this.scopeStack=[],this.numDataMovesStack=[],this.nextScopeId=0,this.tensorInfo=new WeakMap,this.profiling=!1,this.activeProfile={newBytes:0,newTensors:0,peakBytes:0,kernels:[],result:null,get kernelNames(){return Array.from(new Set(this.kernels.map(t=>t.name)))}}}dispose(){for(const t in this.registeredVariables)this.registeredVariables[t].dispose()}}class je{constructor(t){this.ENV=t,this.registry={},this.registryFactory={},this.pendingBackendInitId=0,this.state=new Za}async ready(){if(this.pendingBackendInit!=null)return this.pendingBackendInit.then(()=>{});if(this.backendInstance!=null)return;const t=this.getSortedBackends();for(let n=0;n<t.length;n++){const r=t[n];if(await this.initializeBackend(r).success){await this.setBackend(r);return}}throw new Error("Could not initialize any backends, all backend initializations failed.")}get backend(){if(this.pendingBackendInit!=null)throw new Error(`Backend '${this.backendName}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);if(this.backendInstance==null){const{name:t,asyncInit:n}=this.initializeBackendsAndReturnBest();if(n)throw new Error(`The highest priority backend '${t}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);this.setBackend(t)}return this.backendInstance}backendNames(){return Object.keys(this.registryFactory)}findBackend(t){if(!(t in this.registry))if(t in this.registryFactory){const{asyncInit:n}=this.initializeBackend(t);if(n)return null}else return null;return this.registry[t]}findBackendFactory(t){return t in this.registryFactory?this.registryFactory[t].factory:null}registerBackend(t,n,r=1){return t in this.registryFactory?(se(`${t} backend was already registered. Reusing existing backend factory.`),!1):(this.registryFactory[t]={factory:n,priority:r},!0)}async setBackend(t){if(this.registryFactory[t]==null)throw new Error(`Backend name '${t}' not found in registry`);if(this.backendName=t,this.registry[t]==null){this.backendInstance=null;const{success:n,asyncInit:r}=this.initializeBackend(t);if(!(r?await n:n))return!1}return this.backendInstance=this.registry[t],this.setupRegisteredKernels(),this.profiler=new Ld(this.backendInstance),!0}setupRegisteredKernels(){pr(this.backendName).forEach(n=>{n.setupFunc!=null&&n.setupFunc(this.backendInstance)})}disposeRegisteredKernels(t){pr(t).forEach(r=>{r.disposeFunc!=null&&r.disposeFunc(this.registry[t])})}initializeBackend(t){const n=this.registryFactory[t];if(n==null)throw new Error(`Cannot initialize backend ${t}, no registration found.`);try{const r=n.factory();if(r&&!(r instanceof Uo)&&typeof r.then=="function"){const s=++this.pendingBackendInitId,a=r.then(o=>s<this.pendingBackendInitId?!1:(this.registry[t]=o,this.pendingBackendInit=null,!0)).catch(o=>(s<this.pendingBackendInitId||(this.pendingBackendInit=null,se(`Initialization of backend ${t} failed`),se(o.stack||o.message)),!1));return this.pendingBackendInit=a,{success:a,asyncInit:!0}}else return this.registry[t]=r,{success:!0,asyncInit:!1}}catch(r){return se(`Initialization of backend ${t} failed`),se(r.stack||r.message),{success:!1,asyncInit:!1}}}removeBackend(t){if(!(t in this.registryFactory))throw new Error(`${t} backend not found in registry`);this.backendName===t&&this.pendingBackendInit!=null&&this.pendingBackendInitId++,t in this.registry&&(this.disposeRegisteredKernels(t),this.registry[t].dispose(),delete this.registry[t]),delete this.registryFactory[t],this.backendName===t&&(this.pendingBackendInit=null,this.backendName=null,this.backendInstance=null)}getSortedBackends(){if(Object.keys(this.registryFactory).length===0)throw new Error("No backend found in registry.");return Object.keys(this.registryFactory).sort((t,n)=>this.registryFactory[n].priority-this.registryFactory[t].priority)}initializeBackendsAndReturnBest(){const t=this.getSortedBackends();for(let n=0;n<t.length;n++){const r=t[n],{success:s,asyncInit:a}=this.initializeBackend(r);if(a||s)return{name:r,asyncInit:a}}throw new Error("Could not initialize any backends, all backend initializations failed.")}moveData(t,n){const r=this.state.tensorInfo.get(n),s=r.backend,a=this.readSync(n),o=s.refCount(n);s.disposeData(n,!0),r.backend=t,t.move(n,a,r.shape,r.dtype,o),this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack[this.state.numDataMovesStack.length-1]++}tidy(t,n){let r=null;if(n==null){if(typeof t!="function")throw new Error("Please provide a function to tidy()");n=t}else{if(typeof t!="string"&&!(t instanceof String))throw new Error("When calling with two arguments, the first argument to tidy() must be a string");if(typeof n!="function")throw new Error("When calling with two arguments, the 2nd argument to tidy() must be a function");r=t}let s;return this.scopedRun(()=>this.startScope(r),()=>this.endScope(s),()=>(s=n(),s instanceof Promise&&console.error("Cannot return a Promise inside of tidy."),s))}scopedRun(t,n,r){t();try{const s=r();return n(),s}catch(s){throw n(),s}}nextTensorId(){return je.nextTensorId++}nextVariableId(){return je.nextVariableId++}clone(t){const n=S.runKernel(Ls,{x:t}),r={x:t},s=o=>({x:()=>{const i="float32",u={x:o},l={dtype:i};return S.runKernel(Ps,u,l)}}),a=[];return this.addTapeNode(this.state.activeScope.name,r,[n],s,a,{}),n}runKernel(t,n,r){if(this.backendName==null&&this.backend,!(fn(t,this.backendName)!=null))throw new Error(`Kernel '${t}' not registered for backend '${this.backendName}'`);return this.runKernelFunc({kernelName:t,inputs:n,attrs:r})}shouldCheckForMemLeaks(){return this.ENV.getBool("IS_TEST")}checkKernelForMemLeak(t,n,r){const s=this.backend.numDataIds();let a=0;r.forEach(u=>{a+=u.dtype==="complex64"?3:1});const o=this.state.numDataMovesStack[this.state.numDataMovesStack.length-1],i=s-n-a-o;if(i>0)throw new Error(`Backend '${this.backendName}' has an internal memory leak (${i} data ids) after running '${t}'`)}runKernelFunc(t){let n,r=[];const s=this.isTapeOn(),a=this.state.numBytes,o=this.state.numTensors;this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack.push(0);let i;this.backendName==null&&this.backend;let u;const l=Wr(t)?t.kernelName:this.state.activeScope!=null?this.state.activeScope.name:"";if(Wr(t)){const{kernelName:g,inputs:N,attrs:w}=t;this.backendName==null&&this.backend;const T=fn(g,this.backendName);y(T!=null,()=>`Cannot find registered kernel '${g}' for backend '${this.backendName}'`),i=()=>{const x=this.backend.numDataIds();u=T.kernelFunc({inputs:N,attrs:w,backend:this.backend});const $=Array.isArray(u)?u:[u];this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(g,x,$);const E=$.map(I=>I.rank!=null?I:this.makeTensorFromTensorInfo(I));if(s){const I=this.getTensorsForGradient(g,N,E);r=this.saveTensorsForBackwardMode(I)}return E}}else{const{forwardFunc:g}=t,N=w=>{s&&(r=w.map(T=>this.keep(this.clone(T))))};i=()=>{const w=this.backend.numDataIds();u=this.tidy(()=>g(this.backend,N));const T=Array.isArray(u)?u:[u];return this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(l,w,T),T}}const{inputs:h,attrs:c}=t,p=Wr(t)?null:t.backwardsFunc;let d;return this.scopedRun(()=>this.state.kernelDepth++,()=>this.state.kernelDepth--,()=>{!this.ENV.getBool("DEBUG")&&!this.state.profiling?n=i():(d=this.profiler.profileKernel(l,h,()=>i()),this.ENV.getBool("DEBUG")&&this.profiler.logKernelProfile(d),n=d.outputs)}),s&&this.addTapeNode(l,h,n,p,r,c),this.state.profiling&&this.state.activeProfile.kernels.push({name:l,bytesAdded:this.state.numBytes-a,totalBytesSnapshot:this.state.numBytes,tensorsAdded:this.state.numTensors-o,totalTensorsSnapshot:this.state.numTensors,inputShapes:Object.keys(h).map(g=>h[g]!=null?h[g].shape:null),outputShapes:n.map(g=>g.shape),kernelTimeMs:d.timeMs,extraInfo:d.extraInfo}),Array.isArray(u)?n:n[0]}saveTensorsForBackwardMode(t){return t.map(r=>this.keep(this.clone(r)))}getTensorsForGradient(t,n,r){const s=ns(t);if(s!=null){const a=s.inputsToSave||[],o=s.outputsToSave||[];let i;s.saveAllInputs?(y(Array.isArray(n),()=>"saveAllInputs is true, expected inputs to be an array."),i=Object.keys(n).map(l=>n[l])):i=a.map(l=>n[l]);const u=r.filter((l,h)=>o[h]);return i.concat(u)}return[]}makeTensor(t,n,r,s){if(t==null)throw new Error("Values passed to engine.makeTensor() are null");r=r||"float32",s=s||this.backend;let a=t;r==="string"&&oe(t[0])&&(a=t.map(u=>xn(u)));const o=s.write(a,n,r),i=new et(n,r,o,this.nextTensorId());if(this.trackTensor(i,s),r==="string"){const u=this.state.tensorInfo.get(o),l=Yo(a);this.state.numBytes+=l-u.bytes,u.bytes=l}return i}makeTensorFromDataId(t,n,r,s){r=r||"float32";const a={dataId:t,shape:n,dtype:r};return this.makeTensorFromTensorInfo(a,s)}makeTensorFromTensorInfo(t,n){const{dataId:r,shape:s,dtype:a}=t,o=new et(s,a,r,this.nextTensorId());return this.trackTensor(o,n),o}makeVariable(t,n=!0,r,s){r=r||this.nextVariableId().toString(),s!=null&&s!==t.dtype&&(t=t.cast(s));const a=new mn(t,n,r,this.nextTensorId());if(this.state.registeredVariables[a.name]!=null)throw new Error(`Variable with name ${a.name} was already registered`);return this.state.registeredVariables[a.name]=a,this.incRef(a,this.backend),a}trackTensor(t,n){this.state.numTensors++,t.dtype==="string"&&this.state.numStringTensors++;let r=0;t.dtype!=="complex64"&&t.dtype!=="string"&&(r=t.size*cr(t.dtype)),this.state.numBytes+=r,this.state.tensorInfo.has(t.dataId)||(this.state.numDataBuffers++,this.state.tensorInfo.set(t.dataId,{backend:n||this.backend,dtype:t.dtype,shape:t.shape,bytes:r})),t instanceof mn||this.track(t)}incRef(t,n){this.trackTensor(t,n),this.backend.incRef(t.dataId)}removeDataId(t,n){this.state.tensorInfo.has(t)&&this.state.tensorInfo.get(t).backend===n&&(this.state.tensorInfo.delete(t),this.state.numDataBuffers--)}disposeTensor(t){if(!this.state.tensorInfo.has(t.dataId))return;const n=this.state.tensorInfo.get(t.dataId);if(this.state.numTensors--,t.dtype==="string"&&(this.state.numStringTensors--,this.state.numBytes-=n.bytes),t.dtype!=="complex64"&&t.dtype!=="string"){const r=t.size*cr(t.dtype);this.state.numBytes-=r}n.backend.disposeData(t.dataId)&&this.removeDataId(t.dataId,n.backend)}disposeVariables(){for(const t in this.state.registeredVariables){const n=this.state.registeredVariables[t];this.disposeVariable(n)}}disposeVariable(t){this.disposeTensor(t),this.state.registeredVariables[t.name]!=null&&delete this.state.registeredVariables[t.name]}memory(){const t=this.backend.memory();return t.numTensors=this.state.numTensors,t.numDataBuffers=this.state.numDataBuffers,t.numBytes=this.state.numBytes,this.state.numStringTensors>0&&(t.unreliable=!0,t.reasons==null&&(t.reasons=[]),t.reasons.push("Memory usage by string tensors is approximate (2 bytes per character)")),t}async profile(t){this.state.profiling=!0;const n=this.state.numBytes,r=this.state.numTensors;this.state.activeProfile.kernels=[],this.state.activeProfile.result=await t(),this.state.profiling=!1,this.state.activeProfile.peakBytes=Math.max(...this.state.activeProfile.kernels.map(s=>s.totalBytesSnapshot)),this.state.activeProfile.newBytes=this.state.numBytes-n,this.state.activeProfile.newTensors=this.state.numTensors-r;for(const s of this.state.activeProfile.kernels)s.kernelTimeMs=await s.kernelTimeMs,s.extraInfo=await s.extraInfo;return this.state.activeProfile}isTapeOn(){return this.state.gradientDepth>0&&this.state.kernelDepth===0}addTapeNode(t,n,r,s,a,o){const i={id:this.state.nextTapeNodeId++,kernelName:t,inputs:n,outputs:r,saved:a},u=ns(t);u!=null&&(s=u.gradFunc),s!=null&&(i.gradient=l=>(l=l.map((h,c)=>{if(h==null){const p=r[c],d=vr(p.size,p.dtype);return this.makeTensor(d,p.shape,p.dtype)}return h}),s(l.length>1?l:l[0],a,o))),this.state.activeTape.push(i)}keep(t){return t.kept=!0,t}startTape(){this.state.gradientDepth===0&&(this.state.activeTape=[]),this.state.gradientDepth++}endTape(){this.state.gradientDepth--}startScope(t){const n={track:[],name:"unnamed scope",id:this.state.nextScopeId++};t&&(n.name=t),this.state.scopeStack.push(n),this.state.activeScope=n}endScope(t){const n=Vs(t),r=new Set(n.map(a=>a.id));for(let a=0;a<this.state.activeScope.track.length;a++){const o=this.state.activeScope.track[a];!o.kept&&!r.has(o.id)&&o.dispose()}const s=this.state.scopeStack.pop();this.state.activeScope=this.state.scopeStack.length===0?null:this.state.scopeStack[this.state.scopeStack.length-1],n.forEach(a=>{!a.kept&&a.scopeId===s.id&&this.track(a)})}gradients(t,n,r,s=!1){if(y(n.length>0,()=>"gradients() received an empty list of xs."),r!=null&&r.dtype!=="float32")throw new Error(`dy must have 'float32' dtype, but has '${r.dtype}'`);const a=this.scopedRun(()=>this.startTape(),()=>this.endTape(),()=>this.tidy("forward",t));y(a instanceof et,()=>"The result y returned by f() must be a tensor.");const o=Vd(this.state.activeTape,n,a);if(!s&&o.length===0&&n.length>0)throw new Error("Cannot compute gradient of y=f(x) with respect to x. Make sure that the f you passed encloses all operations that lead from x to y.");return this.tidy("backward",()=>{const i={};i[a.id]=r??Jd(a.shape),Wd(i,o,l=>this.tidy(l),Qd);const u=n.map(l=>i[l.id]);return this.state.gradientDepth===0&&(this.state.activeTape.forEach(l=>{for(const h of l.saved)h.dispose()}),this.state.activeTape=null),{value:a,grads:u}})}customGrad(t){return y(ce(t),()=>"The f passed in customGrad(f) must be a function."),(...n)=>{y(n.every(i=>i instanceof et),()=>"The args passed in customGrad(f)(x1, x2,...) must all be tensors");let r;const s={};n.forEach((i,u)=>{s[u]=i});const a=(i,u)=>(r=t(...n,u),y(r.value instanceof et,()=>"The function f passed in customGrad(f) must return an object where `obj.value` is a tensor"),y(ce(r.gradFunc),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function."),r.value),o=(i,u)=>{const l=r.gradFunc(i,u),h=Array.isArray(l)?l:[l];y(h.length===n.length,()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns the same number of tensors as inputs passed to f(...)."),y(h.every(p=>p instanceof et),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns a list of only tensors.");const c={};return h.forEach((p,d)=>{c[d]=()=>p}),c};return this.runKernelFunc({forwardFunc:a,backwardsFunc:o,inputs:s})}}readSync(t){return this.state.tensorInfo.get(t).backend.readSync(t)}read(t){return this.state.tensorInfo.get(t).backend.read(t)}readToGPU(t,n){return this.state.tensorInfo.get(t).backend.readToGPU(t,n)}async time(t){const n=dn(),r=await this.backend.time(t);return r.wallMs=dn()-n,r}track(t){return this.state.activeScope!=null&&(t.scopeId=this.state.activeScope.id,this.state.activeScope.track.push(t)),t}get registeredVariables(){return this.state.registeredVariables}reset(){this.pendingBackendInitId++,this.state.dispose(),this.ENV.reset(),this.state=new Za;for(const t in this.registry)this.disposeRegisteredKernels(t),this.registry[t].dispose(),delete this.registry[t];this.backendName=null,this.backendInstance=null,this.pendingBackendInit=null}}je.nextTensorId=0;je.nextVariableId=0;function Jd(e){const t=Ds(K(e),"float32");return S.makeTensor(t,e,"float32")}function rc(){const e=ei();if(e._tfengine==null){const t=new ti(e);e._tfengine=new je(t)}return Qf(e._tfengine.ENV),Hd(()=>e._tfengine),e._tfengine}const S=rc();function Qd(e,t){const n={a:e,b:t};return S.runKernel(Bs,n)}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function tm(){return typeof navigator<"u"&&navigator!=null}let ls;function em(e){ls=e}function nm(e){if(ls!==void 0)return ls;if(e||tm()){if(e||(e=navigator),e.product==="ReactNative")return!0;const t=e.userAgent||e.vendor||(typeof window<"u"?window.opera:"");if(!t){const n=e;return n.userAgentData&&n.userAgentData.mobile}return/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino/i.test(t)||/1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(t.substr(0,4))}return!1}function sc(){return typeof window<"u"&&window.document!=null||typeof WorkerGlobalScope<"u"}const rm=Object.freeze(Object.defineProperty({__proto__:null,isBrowser:sc,isMobile:nm,mockIsMobile:em},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const bt=M();bt.registerFlag("DEBUG",()=>!1,e=>{e&&console.warn("Debugging mode is ON. The output of every math call will be downloaded to CPU and checked for NaNs. This significantly impacts performance.")});bt.registerFlag("IS_BROWSER",()=>sc());bt.registerFlag("IS_NODE",()=>typeof process<"u"&&typeof process.versions<"u"&&typeof process.versions.node<"u");bt.registerFlag("IS_CHROME",()=>typeof navigator<"u"&&navigator!=null&&navigator.userAgent!=null&&/Chrome/.test(navigator.userAgent)&&/Google Inc/.test(navigator.vendor));bt.registerFlag("IS_SAFARI",()=>typeof navigator<"u"&&navigator!=null&&navigator.userAgent!=null&&/Safari/.test(navigator.userAgent)&&/Apple/.test(navigator.vendor));bt.registerFlag("PROD",()=>!1);bt.registerFlag("TENSORLIKE_CHECK_SHAPE_CONSISTENCY",()=>bt.getBool("DEBUG"));bt.registerFlag("DEPRECATION_WARNINGS_ENABLED",()=>!0);bt.registerFlag("IS_TEST",()=>!1);bt.registerFlag("CHECK_COMPUTATION_FOR_ERRORS",()=>bt.getBool("DEBUG"));bt.registerFlag("WRAP_TO_IMAGEBITMAP",()=>!1);bt.registerFlag("CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU",()=>!1);bt.registerFlag("USE_SETTIMEOUTCUSTOM",()=>!1);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ut(e,t){let n=e;if(ut(e))return t==="string"?[]:[e.length];if(Ql(e)){const s=e.channels||"RGBA";return[e.height,e.width*s.length]}else if(tc(e))return[e.buffer.size/(t==null?4:cr(t))];if(!Array.isArray(e))return[];const r=[];for(;Array.isArray(n)||ut(n)&&t!=="string";)r.push(n.length),n=n[0];return Array.isArray(e)&&M().getBool("TENSORLIKE_CHECK_SHAPE_CONSISTENCY")&&ac(e,r,[]),r}function ac(e,t,n){if(n=n||[],!Array.isArray(e)&&!ut(e)){y(t.length===0,()=>`Element arr[${n.join("][")}] is a primitive, but should be an array/TypedArray of ${t[0]} elements`);return}y(t.length>0,()=>`Element arr[${n.join("][")}] should be a primitive, but is an array of ${e.length} elements`),y(e.length===t[0],()=>`Element arr[${n.join("][")}] should have ${t[0]} elements, but has ${e.length} elements`);const r=t.slice(1);for(let s=0;s<e.length;++s)ac(e[s],r,n.concat(s))}function Ja(e,t,n,r){if(e!=="string_or_numeric"){if(e==null)throw new Error("Expected dtype cannot be null.");if(e!=="numeric"&&e!==t||e==="numeric"&&t==="string")throw new Error(`Argument '${n}' passed to '${r}' must be ${e} tensor, but got ${t} tensor`)}}function m(e,t,n,r="numeric"){if(e instanceof Jl())return Ja(r,e.dtype,t,n),e;let s=kn(e);if(s!=="string"&&["bool","int32","float32"].indexOf(r)>=0&&(s=r),Ja(r,s,t,n),e==null||!ut(e)&&!Array.isArray(e)&&typeof e!="number"&&typeof e!="boolean"&&typeof e!="string"){const u=e==null?"null":e.constructor.name;throw new Error(`Argument '${t}' passed to '${n}' must be a Tensor or TensorLike, but got '${u}'`)}const a=Ut(e,s);!ut(e)&&!Array.isArray(e)&&(e=[e]);const i=s!=="string"?Sr(e,s):pe(e,[],!0);return S.makeTensor(i,a,s)}function gn(e,t,n,r="numeric"){if(!Array.isArray(e))throw new Error(`Argument ${t} passed to ${n} must be a \`Tensor[]\` or \`TensorLike[]\``);return e.map((a,o)=>m(a,`${t}[${o}]`,n,r))}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ws="__op";function v(e){const t=Object.keys(e);if(t.length!==1)throw new Error(`Please provide an object with a single key (operation name) mapping to a function. Got an object with ${t.length} keys.`);let n=t[0];const r=e[n];n.endsWith("_")&&(n=n.substring(0,n.length-1)),n=n+Ws;const s=(...a)=>{S.startScope(n);try{const o=r(...a);return he(o)&&console.error("Cannot return a Promise inside of tidy."),S.endScope(o),o}catch(o){throw S.endScope(null),o}};return Object.defineProperty(s,"name",{value:n,configurable:!0}),s}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function sm(e,t){const n=m(e,"real","complex"),r=m(t,"imag","complex");gt(n.shape,r.shape,`real and imag shapes, ${n.shape} and ${r.shape}, must match in call to tf.complex().`);const s={real:n,imag:r};return S.runKernel(Ei,s)}const ee=v({complex_:sm});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ye(e,t,n,r){if(r==null)r=kn(e);else if(r==="complex64")throw new Error("Cannot construct a complex64 tensor directly. Please use tf.complex(real, imag).");if(tc(e)||Ql(e)){if(r!=="float32"&&r!=="int32")throw new Error(`Creating tensor from GPU data only supports 'float32'|'int32' dtype, while the dtype is ${r}.`);return S.backend.createTensorFromGPUData(e,t||n,r)}if(!ut(e)&&!Array.isArray(e)&&typeof e!="number"&&typeof e!="boolean"&&typeof e!="string")throw new Error("values passed to tensor(values) must be a number/boolean/string or an array of numbers/booleans/strings, or a TypedArray");if(t!=null){$t(t);const s=K(t),a=K(n);y(s===a,()=>`Based on the provided shape, [${t}], the tensor should have ${s} values but has ${a}`);for(let o=0;o<n.length;++o){const i=n[o],u=o===n.length-1?i!==K(t.slice(o)):!0;y(n[o]===t[o]||!u,()=>`Error creating a new Tensor. Inferred shape (${n}) does not match the provided shape (${t}). `)}}return!ut(e)&&!Array.isArray(e)&&(e=[e]),t=t||n,e=r!=="string"?Sr(e,r):pe(e,[],!0),S.makeTensor(e,t,r)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xt(e,t,n){const r=Ut(e,n);return ye(e,t,r,n)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ie={float32:4,float16:2,int32:4,uint16:2,uint8:1,bool:1,complex64:8};class Ct{static join(t){return new Ct(t).slice()}constructor(t){if(this.shards=[],this.previousShardIndex=0,t==null||(t instanceof Array||(t=[t]),t=t.map(r=>ut(r)?r.buffer:r),t.length===0))return;this.bufferUniformSize=t[0].byteLength;let n=0;for(let r=0;r<t.length;r++){const s=t[r];r!==t.length-1&&s.byteLength!==this.bufferUniformSize&&(this.bufferUniformSize=void 0);const a=n+s.byteLength;this.shards.push({buffer:s,start:n,end:a}),n=a}this.shards.length===0&&(this.byteLength=0),this.byteLength=this.shards[this.shards.length-1].end}slice(t=0,n=this.byteLength){if(this.shards.length===0)return new ArrayBuffer(0);if(t=isNaN(Number(t))?0:t,n=isNaN(Number(n))?0:n,t=Math.max(0,t),n=Math.min(this.byteLength,n),n<=t)return new ArrayBuffer(0);const r=this.findShardForByte(t);if(r===-1)throw new Error(`Could not find start shard for byte ${t}`);const s=n-t,a=new ArrayBuffer(s),o=new Uint8Array(a);let i=0;for(let u=r;u<this.shards.length;u++){const l=this.shards[u],c=t+i-l.start,p=i,g=Math.min(n,l.end)-l.start,N=new Uint8Array(l.buffer,c,g-c);if(o.set(N,p),i+=N.length,n<l.end)break}return a}findShardForByte(t){if(this.shards.length===0||t<0||t>=this.byteLength)return-1;if(this.bufferUniformSize!=null)return this.previousShardIndex=Math.floor(t/this.bufferUniformSize),this.previousShardIndex;function n(s){return t<s.start?-1:t>=s.end?1:0}if(n(this.shards[this.previousShardIndex])===0)return this.previousShardIndex;const r=am(this.shards,n);return r===-1?-1:(this.previousShardIndex=r,this.previousShardIndex)}}function am(e,t){let n=0,r=e.length;for(;n<=r;){const s=Math.floor((r-n)/2)+n,a=t(e[s]);if(a===0)return s;a<0?r=s:n=s+1}return-1}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function om(){M().set("PROD",!0)}function im(){M().set("DEBUG",!0)}function um(){M().set("DEPRECATION_WARNINGS_ENABLED",!1),console.warn("TensorFlow.js deprecation warnings have been disabled.")}function lm(e){M().getBool("DEPRECATION_WARNINGS_ENABLED")&&console.warn(e+" You can disable deprecation warnings with tf.disableDeprecationWarnings().")}function cm(){S.disposeVariables()}function Us(){return S}function hm(){return S.memory()}function pm(e){return S.profile(e)}function V(e,t){return S.tidy(e,t)}function mt(e){Vs(e).forEach(n=>n.dispose())}function Rt(e){return S.keep(e)}function fm(e){return S.time(e)}function dm(e){return S.setBackend(e)}function mm(){return S.ready()}function qs(){return S.backendName}function gm(e){S.removeBackend(e)}function ym(e){return S.findBackend(e)}function bm(e){return S.findBackendFactory(e)}function wm(e,t,n=1){return S.registerBackend(e,t,n)}function Hs(){return S.backend}function Nm(e,t){M().setPlatform(e,t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const fe=4;async function vm(e,t){const n=[],r=[],s=Array.isArray(e)?e.map(o=>o.name):Object.keys(e);for(let o=0;o<s.length;++o){const i=s[o],u=Array.isArray(e)?e[o].tensor:e[i];if(u.dtype!=="float32"&&u.dtype!=="int32"&&u.dtype!=="bool"&&u.dtype!=="string"&&u.dtype!=="complex64")throw new Error(`Unsupported dtype in weight '${i}': ${u.dtype}`);const l={name:i,shape:u.shape,dtype:u.dtype};if(u.dtype==="string"){const h=new Promise(async c=>{const p=await u.bytes(),d=p.reduce((w,T)=>w+T.length,0)+fe*p.length,g=new Uint8Array(d);let N=0;for(let w=0;w<p.length;w++){const T=p[w],x=new Uint8Array(new Uint32Array([T.length]).buffer);g.set(x,N),N+=fe,g.set(T,N),N+=T.length}c(g)});r.push(h)}else r.push(u.data());t!=null&&(l.group=t),n.push(l)}const a=await Promise.all(r);return{data:Em(a),specs:n}}function oc(e,t){const n=new Ct(e),r={};let s=0;for(const a of t){const o=Sm(a,(i,u)=>n.slice(s+i,s+u));r[a.name]=ic(a,n.slice(s,s+o)),s+=o}return r}function Sm(e,t){const n=K(e.shape);let r;if("quantization"in e){const s=e.quantization;r=Ie[s.dtype]}else if(e.dtype==="string"){let s=0;for(let a=0;a<n;a++)s+=fe+new Uint32Array(t(s,s+fe))[0];return s}else r=Ie[e.dtype];return n*r}async function Tm(e,t){const n=K(e.shape);let r;if("quantization"in e){const s=e.quantization;r=Ie[s.dtype]}else if(e.dtype==="string"){let s=0;for(let a=0;a<n;a++)s+=fe+new Uint32Array(await t(s,s+fe))[0];return s}else r=Ie[e.dtype];return n*r}function ic(e,t){const n=e.name,r=e.dtype,s=e.shape,a=K(s);let o,i=0;if("quantization"in e){const u=e.quantization;if(u.dtype==="uint8"||u.dtype==="uint16"){if(!("min"in u&&"scale"in u))throw new Error(`Weight ${e.name} with quantization ${u.dtype} doesn't have corresponding metadata min and scale.`)}else if(u.dtype==="float16"){if(r!=="float32")throw new Error(`Weight ${e.name} is quantized with ${u.dtype} which only supports weights of type float32 not ${r}.`)}else throw new Error(`Weight ${e.name} has unknown quantization dtype ${u.dtype}. Supported quantization dtypes are: 'uint8', 'uint16', and 'float16'.`);const l=Ie[u.dtype],h=u.dtype==="uint8"?new Uint8Array(t):new Uint16Array(t);if(r==="float32")if(u.dtype==="uint8"||u.dtype==="uint16"){o=new Float32Array(h.length);for(let c=0;c<h.length;c++){const p=h[c];o[c]=p*u.scale+u.min}}else if(u.dtype==="float16")o=Om()(h);else throw new Error(`Unsupported quantization type ${u.dtype} for weight type float32.`);else if(r==="int32"){if(u.dtype!=="uint8"&&u.dtype!=="uint16")throw new Error(`Unsupported quantization type ${u.dtype} for weight type int32.`);o=new Int32Array(h.length);for(let c=0;c<h.length;c++){const p=h[c];o[c]=Math.round(p*u.scale+u.min)}}else throw new Error(`Unsupported dtype in weight '${n}': ${r}`);i+=a*l}else if(r==="string"){const u=K(e.shape);o=[];for(let l=0;l<u;l++){const h=new Uint32Array(t.slice(i,i+fe))[0];i+=fe;const c=new Uint8Array(t.slice(i,i+h));o.push(c),i+=h}}else{const u=Ie[r];if(r==="float32")o=new Float32Array(t);else if(r==="int32")o=new Int32Array(t);else if(r==="bool")o=new Uint8Array(t);else if(r==="complex64"){o=new Float32Array(t);const l=new Float32Array(o.length/2),h=new Float32Array(o.length/2);for(let g=0;g<l.length;g++)l[g]=o[g*2],h[g]=o[g*2+1];const c=xt(l,s,"float32"),p=xt(h,s,"float32"),d=ee(c,p);return c.dispose(),p.dispose(),d}else throw new Error(`Unsupported dtype in weight '${n}': ${r}`);i+=a*u}return xt(o,s,r)}async function Qa(e,t,n){let r=new Uint8Array(t);for(;r.byteLength<n;){const{done:s,value:a}=await e.read();if(s&&a==null){const i=n-r.byteLength;throw new Error(`Reader is done but ${i} bytes are still expected`)}const o=new Uint8Array(r.length+a.byteLength);o.set(r,0),o.set(new Uint8Array(a),r.length),r=o}return r.buffer}async function uc(e,t){const n={},r=e.getReader();let s=new ArrayBuffer(0);for(const a of t){const o=await Tm(a,async(l,h)=>(s=await Qa(r,s,h),s.slice(l,h)));s=await Qa(r,s,o);const i=s.slice(0,o);s=s.slice(o);const u=ic(a,i);if(n[a.name]=u,qs()==="webgpu"){const l=Hs();"uploadToGPU"in l&&K(u.shape)>=M().get("WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD")&&l.uploadToGPU(u.dataId)}}return n}function Em(e){if(e===null)throw new Error(`Invalid input value: ${JSON.stringify(e)}`);let t=0;const n=[];e.forEach(a=>{if(t+=a.byteLength,n.push(a.byteLength===a.buffer.byteLength?a:new a.constructor(a)),!(a instanceof Float32Array||a instanceof Int32Array||a instanceof Uint8Array))throw new Error(`Unsupported TypedArray subtype: ${a.constructor.name}`)});const r=new Uint8Array(t);let s=0;return n.forEach(a=>{r.set(new Uint8Array(a.buffer),s),s+=a.byteLength}),r.buffer}const js=typeof Buffer<"u"&&(typeof Blob>"u"||typeof atob>"u"||typeof btoa>"u");function to(e){return js?Buffer.byteLength(e,"utf8"):new Blob([e]).size}function $m(e){if(js)return Buffer.from(e).toString("base64");const t=new Uint8Array(e);let n="";for(let r=0,s=t.length;r<s;r++)n+=String.fromCharCode(t[r]);return btoa(n)}function _m(e){if(js){const r=Buffer.from(e,"base64");return r.buffer.slice(r.byteOffset,r.byteOffset+r.byteLength)}const t=atob(e),n=new Uint8Array(t.length);for(let r=0;r<t.length;++r)n.set([t.charCodeAt(r)],r);return n.buffer}function km(e){return Ct.join(e)}function eo(e){const t="/";for(e=e.trim();e.endsWith(t);)e=e.slice(0,e.length-1);const n=e.split(t);return n[n.length-1]}function lc(e,t){const n={modelTopology:e.modelTopology,format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy,weightsManifest:t};return e.signature!=null&&(n.signature=e.signature),e.userDefinedMetadata!=null&&(n.userDefinedMetadata=e.userDefinedMetadata),e.modelInitializer!=null&&(n.modelInitializer=e.modelInitializer),e.initializerSignature!=null&&(n.initializerSignature=e.initializerSignature),e.trainingConfig!=null&&(n.trainingConfig=e.trainingConfig),n}function cc(e,t,n){const r={modelTopology:e.modelTopology,format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy};if(e.trainingConfig!=null&&(r.trainingConfig=e.trainingConfig),e.weightsManifest!=null){if(!t)throw new Error("modelJSON has weightsManifest but weightSpecs is null");if(!n)throw new Error("modelJSON has weightsManifest but weightData is null");r.weightSpecs=t,r.weightData=n}return e.signature!=null&&(r.signature=e.signature),e.userDefinedMetadata!=null&&(r.userDefinedMetadata=e.userDefinedMetadata),e.modelInitializer!=null&&(r.modelInitializer=e.modelInitializer),e.initializerSignature!=null&&(r.initializerSignature=e.initializerSignature),r}async function Gs(e,t){let n,r;return e.weightsManifest!=null&&([n,r]=await t(e.weightsManifest)),cc(e,n,r)}function An(e){if(e.modelTopology instanceof ArrayBuffer)throw new Error("Expected JSON model topology, received ArrayBuffer.");return{dateSaved:new Date,modelTopologyType:"JSON",modelTopologyBytes:e.modelTopology==null?0:to(JSON.stringify(e.modelTopology)),weightSpecsBytes:e.weightSpecs==null?0:to(JSON.stringify(e.weightSpecs)),weightDataBytes:e.weightData==null?0:new Ct(e.weightData).byteLength}}function cs(e){const t=[];for(const n of e)t.push(...n.weights);return t}function Im(){const e=n=>{let r=n<<13,s=0;for(;(r&8388608)===0;)s-=8388608,r<<=1;return r&=-8388609,s+=947912704,r|s},t=new Uint32Array(2048);t[0]=0;for(let n=1;n<1024;n++)t[n]=e(n);for(let n=1024;n<2048;n++)t[n]=939524096+(n-1024<<13);return t}function xm(){const e=new Uint32Array(64);e[0]=0,e[31]=1199570944,e[32]=2147483648,e[63]=3347054592;for(let t=1;t<31;t++)e[t]=t<<23;for(let t=33;t<63;t++)e[t]=2147483648+(t-32<<23);return e}function Am(){const e=new Uint32Array(64);for(let t=0;t<64;t++)e[t]=1024;return e[0]=e[32]=0,e}function Om(){const e=Im(),t=xm(),n=Am();return r=>{const s=new ArrayBuffer(4*r.length),a=new Uint32Array(s);for(let o=0;o<r.length;o++){const i=r[o],u=e[n[i>>10]+(i&1023)]+t[i>>10];a[o]=u}return new Float32Array(s)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class rt{constructor(){this.saveRouters=[],this.loadRouters=[]}static getInstance(){return rt.instance==null&&(rt.instance=new rt),rt.instance}static registerSaveRouter(t){rt.getInstance().saveRouters.push(t)}static registerLoadRouter(t){rt.getInstance().loadRouters.push(t)}static getSaveHandlers(t){return rt.getHandlers(t,"save")}static getLoadHandlers(t,n){return rt.getHandlers(t,"load",n)}static getHandlers(t,n,r){const s=[];return(n==="load"?rt.getInstance().loadRouters:rt.getInstance().saveRouters).forEach(o=>{const i=o(t,r);i!==null&&s.push(i)}),s}}const Dm=e=>rt.registerSaveRouter(e),Fm=e=>rt.registerLoadRouter(e),Rm=e=>rt.getSaveHandlers(e),Bm=(e,t)=>rt.getLoadHandlers(e,t);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const hs="tensorflowjs",ps=1,$e="models_store",ie="model_info_store";function hc(){if(!M().getBool("IS_BROWSER"))throw new Error("Failed to obtain IndexedDB factory because the current environmentis not a web browser.");const e=typeof window>"u"?self:window,t=e.indexedDB||e.mozIndexedDB||e.webkitIndexedDB||e.msIndexedDB||e.shimIndexedDB;if(t==null)throw new Error("The current browser does not appear to support IndexedDB.");return t}function fs(e){const t=e.result;t.createObjectStore($e,{keyPath:"modelPath"}),t.createObjectStore(ie,{keyPath:"modelPath"})}class xe{constructor(t){if(this.indexedDB=hc(),t==null||!t)throw new Error("For IndexedDB, modelPath must not be null, undefined or empty.");this.modelPath=t}async save(t){if(t.modelTopology instanceof ArrayBuffer)throw new Error("BrowserLocalStorage.save() does not support saving model topology in binary formats yet.");return this.databaseAction(this.modelPath,t)}async load(){return this.databaseAction(this.modelPath)}databaseAction(t,n){return new Promise((r,s)=>{const a=this.indexedDB.open(hs,ps);a.onupgradeneeded=()=>fs(a),a.onsuccess=()=>{const o=a.result;if(n==null){const i=o.transaction($e,"readonly"),l=i.objectStore($e).get(this.modelPath);l.onsuccess=()=>{if(l.result==null)return o.close(),s(new Error(`Cannot find model with path '${this.modelPath}' in IndexedDB.`));r(l.result.modelArtifacts)},l.onerror=h=>(o.close(),s(l.error)),i.oncomplete=()=>o.close()}else{n.weightData=Ct.join(n.weightData);const i=An(n),u=o.transaction(ie,"readwrite");let l=u.objectStore(ie),h;try{h=l.put({modelPath:this.modelPath,modelArtifactsInfo:i})}catch(p){return s(p)}let c;h.onsuccess=()=>{c=o.transaction($e,"readwrite");const p=c.objectStore($e);let d;try{d=p.put({modelPath:this.modelPath,modelArtifacts:n,modelArtifactsInfo:i})}catch(g){return s(g)}d.onsuccess=()=>r({modelArtifactsInfo:i}),d.onerror=g=>{l=u.objectStore(ie);const N=l.delete(this.modelPath);N.onsuccess=()=>(o.close(),s(d.error)),N.onerror=w=>(o.close(),s(d.error))}},h.onerror=p=>(o.close(),s(h.error)),u.oncomplete=()=>{c==null?o.close():c.oncomplete=()=>o.close()}}},a.onerror=o=>s(a.error)})}}xe.URL_SCHEME="indexeddb://";const pc=e=>M().getBool("IS_BROWSER")&&!Array.isArray(e)&&e.startsWith(xe.URL_SCHEME)?Pm(e.slice(xe.URL_SCHEME.length)):null;rt.registerSaveRouter(pc);rt.registerLoadRouter(pc);function Pm(e){return new xe(e)}function Cm(e){return e.startsWith(xe.URL_SCHEME)?e.slice(xe.URL_SCHEME.length):e}class Lm{constructor(){this.indexedDB=hc()}async listModels(){return new Promise((t,n)=>{const r=this.indexedDB.open(hs,ps);r.onupgradeneeded=()=>fs(r),r.onsuccess=()=>{const s=r.result,a=s.transaction(ie,"readonly"),i=a.objectStore(ie).getAll();i.onsuccess=()=>{const u={};for(const l of i.result)u[l.modelPath]=l.modelArtifactsInfo;t(u)},i.onerror=u=>(s.close(),n(i.error)),a.oncomplete=()=>s.close()},r.onerror=s=>n(r.error)})}async removeModel(t){return t=Cm(t),new Promise((n,r)=>{const s=this.indexedDB.open(hs,ps);s.onupgradeneeded=()=>fs(s),s.onsuccess=()=>{const a=s.result,o=a.transaction(ie,"readwrite"),i=o.objectStore(ie),u=i.get(t);let l;u.onsuccess=()=>{if(u.result==null)return a.close(),r(new Error(`Cannot find model with path '${t}' in IndexedDB.`));{const h=i.delete(t),c=()=>{l=a.transaction($e,"readwrite");const d=l.objectStore($e).delete(t);d.onsuccess=()=>n(u.result.modelArtifactsInfo),d.onerror=g=>r(u.error)};h.onsuccess=c,h.onerror=p=>(c(),a.close(),r(u.error))}},u.onerror=h=>(a.close(),r(u.error)),o.oncomplete=()=>{l==null?a.close():l.oncomplete=()=>a.close()}},s.onerror=a=>r(s.error)})}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Yt="/",ze="tensorflowjs_models",fc="info",zm="model_topology",Mm="weight_specs",Vm="weight_data",Wm="model_metadata";function dc(e){return{info:[ze,e,fc].join(Yt),topology:[ze,e,zm].join(Yt),weightSpecs:[ze,e,Mm].join(Yt),weightData:[ze,e,Vm].join(Yt),modelMetadata:[ze,e,Wm].join(Yt)}}function mc(e){for(const t of Object.values(e))window.localStorage.removeItem(t)}function Um(e){const t=e.split(Yt);if(t.length<3)throw new Error(`Invalid key format: ${e}`);return t.slice(1,t.length-1).join(Yt)}function qm(e){return e.startsWith(Ae.URL_SCHEME)?e.slice(Ae.URL_SCHEME.length):e}class Ae{constructor(t){if(!M().getBool("IS_BROWSER")||typeof window>"u"||typeof window.localStorage>"u")throw new Error("The current environment does not support local storage.");if(this.LS=window.localStorage,t==null||!t)throw new Error("For local storage, modelPath must not be null, undefined or empty.");this.modelPath=t,this.keys=dc(this.modelPath)}async save(t){if(t.modelTopology instanceof ArrayBuffer)throw new Error("BrowserLocalStorage.save() does not support saving model topology in binary formats yet.");{const n=JSON.stringify(t.modelTopology),r=JSON.stringify(t.weightSpecs),s=An(t),a=Ct.join(t.weightData);try{this.LS.setItem(this.keys.info,JSON.stringify(s)),this.LS.setItem(this.keys.topology,n),this.LS.setItem(this.keys.weightSpecs,r),this.LS.setItem(this.keys.weightData,$m(a));const o={format:t.format,generatedBy:t.generatedBy,convertedBy:t.convertedBy,signature:t.signature!=null?t.signature:void 0,userDefinedMetadata:t.userDefinedMetadata!=null?t.userDefinedMetadata:void 0,modelInitializer:t.modelInitializer!=null?t.modelInitializer:void 0,initializerSignature:t.initializerSignature!=null?t.initializerSignature:void 0,trainingConfig:t.trainingConfig!=null?t.trainingConfig:void 0};return this.LS.setItem(this.keys.modelMetadata,JSON.stringify(o)),{modelArtifactsInfo:s}}catch{throw mc(this.keys),new Error(`Failed to save model '${this.modelPath}' to local storage: size quota being exceeded is a possible cause of this failure: modelTopologyBytes=${s.modelTopologyBytes}, weightSpecsBytes=${s.weightSpecsBytes}, weightDataBytes=${s.weightDataBytes}.`)}}}async load(){const t=JSON.parse(this.LS.getItem(this.keys.info));if(t==null)throw new Error(`In local storage, there is no model with name '${this.modelPath}'`);if(t.modelTopologyType!=="JSON")throw new Error("BrowserLocalStorage does not support loading non-JSON model topology yet.");const n={},r=JSON.parse(this.LS.getItem(this.keys.topology));if(r==null)throw new Error(`In local storage, the topology of model '${this.modelPath}' is missing.`);n.modelTopology=r;const s=JSON.parse(this.LS.getItem(this.keys.weightSpecs));if(s==null)throw new Error(`In local storage, the weight specs of model '${this.modelPath}' are missing.`);n.weightSpecs=s;const a=this.LS.getItem(this.keys.modelMetadata);if(a!=null){const i=JSON.parse(a);n.format=i.format,n.generatedBy=i.generatedBy,n.convertedBy=i.convertedBy,i.signature!=null&&(n.signature=i.signature),i.userDefinedMetadata!=null&&(n.userDefinedMetadata=i.userDefinedMetadata),i.modelInitializer!=null&&(n.modelInitializer=i.modelInitializer),i.initializerSignature!=null&&(n.initializerSignature=i.initializerSignature),i.trainingConfig!=null&&(n.trainingConfig=i.trainingConfig)}const o=this.LS.getItem(this.keys.weightData);if(o==null)throw new Error(`In local storage, the binary weight values of model '${this.modelPath}' are missing.`);return n.weightData=_m(o),n}}Ae.URL_SCHEME="localstorage://";const gc=e=>M().getBool("IS_BROWSER")&&!Array.isArray(e)&&e.startsWith(Ae.URL_SCHEME)?Hm(e.slice(Ae.URL_SCHEME.length)):null;rt.registerSaveRouter(gc);rt.registerLoadRouter(gc);function Hm(e){return new Ae(e)}class jm{constructor(){y(M().getBool("IS_BROWSER"),()=>"Current environment is not a web browser"),y(typeof window>"u"||typeof window.localStorage<"u",()=>"Current browser does not appear to support localStorage"),this.LS=window.localStorage}async listModels(){const t={},n=ze+Yt,r=Yt+fc;for(let s=0;s<this.LS.length;++s){const a=this.LS.key(s);if(a.startsWith(n)&&a.endsWith(r)){const o=Um(a);t[o]=JSON.parse(this.LS.getItem(a))}}return t}async removeModel(t){t=qm(t);const n=dc(t);if(this.LS.getItem(n.info)==null)throw new Error(`Cannot find model at path '${t}'`);const r=JSON.parse(this.LS.getItem(n.info));return mc(n),r}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Me="://";class pt{constructor(){this.managers={}}static getInstance(){return pt.instance==null&&(pt.instance=new pt),pt.instance}static registerManager(t,n){y(t!=null,()=>"scheme must not be undefined or null."),t.endsWith(Me)&&(t=t.slice(0,t.indexOf(Me))),y(t.length>0,()=>"scheme must not be an empty string.");const r=pt.getInstance();y(r.managers[t]==null,()=>`A model store manager is already registered for scheme '${t}'.`),r.managers[t]=n}static getManager(t){const n=pt.getInstance().managers[t];if(n==null)throw new Error(`Cannot find model manager for scheme '${t}'`);return n}static getSchemes(){return Object.keys(pt.getInstance().managers)}}function tr(e){if(e.indexOf(Me)===-1)throw new Error(`The url string provided does not contain a scheme. Supported schemes are: ${pt.getSchemes().join(",")}`);return{scheme:e.split(Me)[0],path:e.split(Me)[1]}}async function yc(e,t,n=!1){y(e!==t,()=>`Old path and new path are the same: '${e}'`);const r=rt.getLoadHandlers(e);y(r.length>0,()=>`Copying failed because no load handler is found for source URL ${e}.`),y(r.length<2,()=>`Copying failed because more than one (${r.length}) load handlers for source URL ${e}.`);const s=r[0],a=rt.getSaveHandlers(t);y(a.length>0,()=>`Copying failed because no save handler is found for destination URL ${t}.`),y(a.length<2,()=>`Copying failed because more than one (${r.length}) save handlers for destination URL ${t}.`);const o=a[0],i=tr(e).scheme,u=tr(e).path,l=i===tr(e).scheme,h=await s.load();n&&l&&await pt.getManager(i).removeModel(u);const c=await o.save(h);return n&&!l&&await pt.getManager(i).removeModel(u),c.modelArtifactsInfo}async function Gm(){const e=pt.getSchemes(),t={};for(const n of e){const r=await pt.getManager(n).listModels();for(const s in r){const a=n+Me+s;t[a]=r[s]}}return t}async function Km(e){const t=tr(e);return pt.getManager(t.scheme).removeModel(t.path)}async function Xm(e,t){return yc(e,t,!1)}async function Ym(e,t){return yc(e,t,!0)}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Zm{constructor(){this.messageName="setTimeoutCustom",this.functionRefs=[],this.handledMessageCount=0,this.hasEventListener=!1}fetch(t,n){return fetch(t,n)}now(){return performance.now()}encode(t,n){if(n!=="utf-8"&&n!=="utf8")throw new Error(`Browser's encoder only supports utf-8, but got ${n}`);return this.textEncoder==null&&(this.textEncoder=new TextEncoder),this.textEncoder.encode(t)}decode(t,n){return new TextDecoder(n).decode(t)}setTimeoutCustom(t,n){if(typeof window>"u"||!M().getBool("USE_SETTIMEOUTCUSTOM")){setTimeout(t,n);return}this.functionRefs.push(t),setTimeout(()=>{window.postMessage({name:this.messageName,index:this.functionRefs.length-1},"*")},n),this.hasEventListener||(this.hasEventListener=!0,window.addEventListener("message",r=>{if(r.source===window&&r.data.name===this.messageName){r.stopPropagation();const s=this.functionRefs[r.data.index];s(),this.handledMessageCount++,this.handledMessageCount===this.functionRefs.length&&(this.functionRefs=[],this.handledMessageCount=0)}},!0))}isTypedArray(t){return jl(t)}}if(M().get("IS_BROWSER")){M().setPlatform("browser",new Zm);try{pt.registerManager(Ae.URL_SCHEME,new jm)}catch{}try{pt.registerManager(xe.URL_SCHEME,new Lm)}catch{}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Jm={importFetch:()=>require("node-fetch")};let Ur;class Qm{constructor(){this.util=require("util"),this.textEncoder=new this.util.TextEncoder}fetch(t,n){return M().global.fetch!=null?M().global.fetch(t,n):(Ur==null&&(Ur=Jm.importFetch()),Ur(t,n))}now(){const t=process.hrtime();return t[0]*1e3+t[1]/1e6}encode(t,n){if(n!=="utf-8"&&n!=="utf8")throw new Error(`Node built-in encoder only supports utf-8, but got ${n}`);return this.textEncoder.encode(t)}decode(t,n){return t.length===0?"":new this.util.TextDecoder(n).decode(t)}isTypedArray(t){return this.util.types.isFloat32Array(t)||this.util.types.isInt32Array(t)||this.util.types.isUint8Array(t)||this.util.types.isUint8ClampedArray(t)}}M().get("IS_NODE")&&!M().get("IS_BROWSER")&&M().setPlatform("node",new Qm);/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qt(e,t="float32",n){return t=t||"float32",$t(e),new dr(e,t,n)}/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function tg(e,t){const n=m(e,"x","cast");if(!Xo(t))throw new Error(`Failed to cast to unknown dtype ${t}`);if(t==="string"&&n.dtype!=="string"||t!=="string"&&n.dtype==="string")throw new Error("Only strings can be casted to strings");const r={x:n},s={dtype:t};return S.runKernel(Ps,r,s)}const j=v({cast_:tg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function eg(e){const n={x:m(e,"x","clone","string_or_numeric")};return S.runKernel(Ls,n)}const Jt=v({clone_:eg});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ks(e,t=!1){console.log(e.toString(t))}/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */rc();const ng={buffer:qt,cast:j,clone:Jt,print:Ks};jd(ng);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function rg(e,t){let n=m(e,"a","add"),r=m(t,"b","add");[n,r]=st(n,r);const s={a:n,b:r};return S.runKernel(Bs,s)}const z=v({add_:rg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function sg(e,t){let n=m(e,"a","floorDiv"),r=m(t,"b","floorDiv");[n,r]=st(n,r);const s={a:n,b:r};return S.runKernel(nu,s)}const Xs=v({floorDiv_:sg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ag(e,t){let n=m(e,"a","div"),r=m(t,"b","div");if([n,r]=st(n,r),n.dtype==="int32"&&r.dtype==="int32")return Xs(n,r);const s={a:n,b:r},a={};return S.runKernel(qi,s,a)}const Y=v({div_:ag});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function og(e,t){let n=m(e,"a","mul"),r=m(t,"b","mul");[n,r]=st(n,r);const s={a:n,b:r};return S.runKernel(Ru,s)}const B=v({mul_:og});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ig(e){const t=m(e,"x","abs");if(t.dtype==="complex64"){const n={x:t};return S.runKernel($i,n)}else{const n={x:t};return S.runKernel(ni,n)}}const Tt=v({abs_:ig});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ug(e){const n={x:m(e,"x","acos")};return S.runKernel(ri,n)}const bc=v({acos_:ug});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function lg(e){const n={x:m(e,"x","acosh")};return S.runKernel(si,n)}const wc=v({acosh_:lg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function cg(e){y(Array.isArray(e),()=>"The argument passed to tf.addN() must be a list of tensors"),y(e.length>=1,()=>`Must pass at least one tensor to tf.addN(), but got ${e.length}`);const t=e.map((s,a)=>m(s,`tensors${a}`,"addN")),n=t[0];t.forEach(s=>{if(s.dtype!==n.dtype)throw new Error("All tensors passed to tf.addN() must have the same dtype")}),t.forEach(s=>{if(!Pt(s.shape,n.shape))throw new Error("All tensors passed to tf.addN() must have the same shape")});const r=t;return S.runKernel(ai,r)}const Nc=v({addN_:cg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hg(e,t=null,n=!1){const s={x:m(e,"x","all","bool")},a={axis:t,keepDims:n};return S.runKernel(oi,s,a)}const vc=v({all_:hg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function pg(e,t=null,n=!1){const s={x:m(e,"x","any","bool")},a={axis:t,keepDims:n};return S.runKernel(ii,s,a)}const Sc=v({any_:pg});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fg(e,t=0){const r={x:m(e,"x","argMax")},s={axis:t};return S.runKernel(ui,r,s)}const Ys=v({argMax_:fg});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function dg(e,t=0){const r={x:m(e,"x","argMin")},s={axis:t};return S.runKernel(li,r,s)}const Tc=v({argMin_:dg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function mg(e){const n={x:m(e,"x","asin")};return S.runKernel(ci,n)}const Ec=v({asin_:mg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function gg(e){const n={x:m(e,"x","asinh")};return S.runKernel(hi,n)}const $c=v({asinh_:gg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function yg(e){const n={x:m(e,"x","atan")};return S.runKernel(pi,n)}const _c=v({atan_:yg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function bg(e,t){let n=m(e,"a","atan2"),r=m(t,"b","atan2");[n,r]=st(n,r);const s={a:n,b:r};return S.runKernel(di,s)}const kc=v({atan2_:bg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function wg(e){const n={x:m(e,"x","atanh")};return S.runKernel(fi,n)}const Ic=v({atanh_:wg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ng(e,t,n,r,s="NHWC",a){const o=e[3],i=[...t,o],u=Oc(s);return On(e,i,n,a,r,null,null,u)}function xc(e,t,n,r,s,a,o="channelsLast"){const[i,u]=yn(t);let l;if(o==="channelsLast")l=[i,u,e[3],e[3]];else if(o==="channelsFirst")l=[i,u,e[1],e[1]];else throw new Error(`Unknown dataFormat ${o}`);return On(e,l,n,r,s,a,!1,o)}function vg(e,t,n,r,s,a,o="NDHWC"){const[i,u,l]=ds(t);let h,c;if(o==="NDHWC")c="channelsLast",h=[i,u,l,e[4],e[4]];else if(o==="NCDHW")c="channelsFirst",h=[i,u,l,e[1],e[1]];else throw new Error(`Unknown dataFormat ${o}`);return Ac(e,h,n,r,s,!1,c,a)}function On(e,t,n,r,s,a,o=!1,i="channelsLast"){let[u,l,h,c]=[-1,-1,-1,-1];if(i==="channelsLast")[u,l,h,c]=e;else if(i==="channelsFirst")[u,c,l,h]=e;else throw new Error(`Unknown dataFormat ${i}`);const[p,d,,g]=t,[N,w]=yn(n),[T,x]=yn(r),$=Ve(p,T),E=Ve(d,x),{padInfo:I,outHeight:A,outWidth:F}=Eg(s,l,h,N,w,$,E,a,i),R=o?g*c:g;let k;return i==="channelsFirst"?k=[u,R,A,F]:i==="channelsLast"&&(k=[u,A,F,R]),{batchSize:u,dataFormat:i,inHeight:l,inWidth:h,inChannels:c,outHeight:A,outWidth:F,outChannels:R,padInfo:I,strideHeight:N,strideWidth:w,filterHeight:p,filterWidth:d,effectiveFilterHeight:$,effectiveFilterWidth:E,dilationHeight:T,dilationWidth:x,inShape:e,outShape:k,filterShape:t}}function Ac(e,t,n,r,s,a=!1,o="channelsLast",i){let[u,l,h,c,p]=[-1,-1,-1,-1,-1];if(o==="channelsLast")[u,l,h,c,p]=e;else if(o==="channelsFirst")[u,p,l,h,c]=e;else throw new Error(`Unknown dataFormat ${o}`);const[d,g,N,,w]=t,[T,x,$]=ds(n),[E,I,A]=ds(r),F=Ve(d,E),R=Ve(g,I),k=Ve(N,A),{padInfo:_,outDepth:b,outHeight:D,outWidth:P}=$g(s,l,h,c,T,x,$,F,R,k,i),C=a?w*p:w;let L;return o==="channelsFirst"?L=[u,C,b,D,P]:o==="channelsLast"&&(L=[u,b,D,P,C]),{batchSize:u,dataFormat:o,inDepth:l,inHeight:h,inWidth:c,inChannels:p,outDepth:b,outHeight:D,outWidth:P,outChannels:C,padInfo:_,strideDepth:T,strideHeight:x,strideWidth:$,filterDepth:d,filterHeight:g,filterWidth:N,effectiveFilterDepth:F,effectiveFilterHeight:R,effectiveFilterWidth:k,dilationDepth:E,dilationHeight:I,dilationWidth:A,inShape:e,outShape:L,filterShape:t}}function Sg(e,t,n,r,s){r==null&&(r=Zs(e,t,n));const a=e[0],o=e[1],i=bn((a-t+2*r)/n+1,s),u=bn((o-t+2*r)/n+1,s);return[i,u]}function Tg(e,t,n,r,s,a){s==null&&(s=Zs(e,t[0],r[0]));const o=[0,0,0,n];for(let i=0;i<3;i++)e[i]+2*s>=t[i]&&(o[i]=bn((e[i]-t[i]+2*s)/r[i]+1,a));return o}function Zs(e,t,n,r=1){const s=Ve(t,r);return Math.floor((e[0]*(n-1)-n+s)/2)}function yn(e){return typeof e=="number"?[e,e,e]:e.length===2?[e[0],e[1],1]:e}function ds(e){return typeof e=="number"?[e,e,e]:e}function Ve(e,t){return t<=1?e:e+(e-1)*(t-1)}function Eg(e,t,n,r,s,a,o,i,u){let l,h,c;if(typeof e=="number"){l={top:e,bottom:e,left:e,right:e,type:e===0?"VALID":"NUMBER"};const d=Sg([t,n],a,r,e,i);h=d[0],c=d[1]}else if(e==="same"){h=Math.ceil(t/r),c=Math.ceil(n/s);const p=Math.max(0,(h-1)*r+a-t),d=Math.max(0,(c-1)*s+o-n),g=Math.floor(p/2),N=p-g,w=Math.floor(d/2),T=d-w;l={top:g,bottom:N,left:w,right:T,type:"SAME"}}else if(e==="valid")l={top:0,bottom:0,left:0,right:0,type:"VALID"},h=Math.ceil((t-a+1)/r),c=Math.ceil((n-o+1)/s);else if(typeof e=="object"){const p=u==="channelsLast"?e[1][0]:e[2][0],d=u==="channelsLast"?e[1][1]:e[2][1],g=u==="channelsLast"?e[2][0]:e[3][0],N=u==="channelsLast"?e[2][1]:e[3][1];l={top:p,bottom:d,left:g,right:N,type:p===0&&d===0&&g===0&&N===0?"VALID":"EXPLICIT"},h=bn((t-a+p+d)/r+1,i),c=bn((n-o+g+N)/s+1,i)}else throw Error(`Unknown padding parameter: ${e}`);return{padInfo:l,outHeight:h,outWidth:c}}function $g(e,t,n,r,s,a,o,i,u,l,h){let c,p,d,g;if(e==="valid"&&(e=0),typeof e=="number"){c={top:e,bottom:e,left:e,right:e,front:e,back:e,type:e===0?"VALID":"NUMBER"};const w=Tg([t,n,r,1],[i,u,l],1,[s,a,o],e,h);p=w[0],d=w[1],g=w[2]}else if(e==="same"){p=Math.ceil(t/s),d=Math.ceil(n/a),g=Math.ceil(r/o);const N=(p-1)*s+i-t,w=(d-1)*a+u-n,T=(g-1)*o+l-r,x=Math.floor(N/2),$=N-x,E=Math.floor(w/2),I=w-E,A=Math.floor(T/2),F=T-A;c={top:E,bottom:I,left:A,right:F,front:x,back:$,type:"SAME"}}else throw Error(`Unknown padding parameter: ${e}`);return{padInfo:c,outDepth:p,outHeight:d,outWidth:g}}function bn(e,t){if(!t)return Math.trunc(e);switch(t){case"round":return Math.round(e);case"ceil":return Math.ceil(e);case"floor":return Math.floor(e);default:throw new Error(`Unknown roundingMode ${t}`)}}function wn(e){const[t,n,r]=yn(e);return t===1&&n===1&&r===1}function ne(e,t){return wn(e)||wn(t)}function Oe(e){return yn(e).every(t=>t>0)}function Oc(e){if(e==="NHWC")return"channelsLast";if(e==="NCHW")return"channelsFirst";throw new Error(`Unknown dataFormat ${e}`)}function Ot(e,t,n){if(n!=null){if(typeof t=="string")throw Error(`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`);if(typeof t=="number")y(qe(t),()=>`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`);else if(typeof t=="object")t.forEach(r=>{r.forEach(s=>{y(qe(s),()=>`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${s}.`)})});else throw Error(`Error in ${e}: Unknown padding parameter: ${t}`)}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _g(e,t){const r={x:m(e,"x","reshape","string_or_numeric")},s={shape:t};return S.runKernel(tl,r,s)}const O=v({reshape_:_g});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function kg(e,t,n,r,s){const a=m(e,"x","avgPool","float32"),o=1;y(ne(n,o),()=>`Error in avgPool: Either strides or dilations must be 1. Got strides ${n} and dilations '${o}'`);let i=a,u=!1;a.rank===3&&(u=!0,i=O(a,[1,a.shape[0],a.shape[1],a.shape[2]])),y(i.rank===4,()=>`Error in avgPool: x must be rank 4 but got rank ${i.rank}.`),Ot("avgPool",r,s);const l={x:i},h={filterSize:t,strides:n,pad:r,dimRoundingMode:s};let c=S.runKernel(mi,l,h);return c=j(c,a.dtype),u?O(c,[c.shape[1],c.shape[2],c.shape[3]]):c}const Js=v({avgPool_:kg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ig(e,t,n,r,s,a="NDHWC"){const o=m(e,"x","avgPool3d","float32");let i=o,u=!1;o.rank===4&&(u=!0,i=O(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]])),y(i.rank===5,()=>`Error in avgPool3d: x must be rank 5 but got rank ${i.rank}.`),y(a==="NDHWC",()=>`Error in avgPool3d: Only NDHWC is currently supported, but got dataFormat of ${a}`),y(typeof n=="number"&&n>0||Array.isArray(n)&&n[0]>0&&n[1]>0&&n[2]>0,()=>`Error in avgPool3d: Stride must be > 0, but got '${n}'`),Ot("avgPool3d",r,s);const l={x:i},h={filterSize:t,strides:n,pad:r,dimRoundingMode:s,dataFormat:a};let c=S.runKernel(gi,l,h);return c=j(c,i.dtype),u?O(c,[c.shape[1],c.shape[2],c.shape[3],c.shape[4]]):c}const Dc=v({avgPool3d_:Ig});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xg(e,t=0){y(e.length>=1,()=>"Pass at least one tensor to concat");const n=gn(e,"tensors","concat","string_or_numeric");if(n[0].dtype==="complex64"&&n.forEach(a=>{if(a.dtype!=="complex64")throw new Error(`Cannot concatenate complex64 tensors with a tensor
          with dtype ${a.dtype}. `)}),n.length===1)return Jt(n[0]);const r=n,s={axis:t};return S.runKernel(_i,r,s)}const ht=v({concat_:xg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ag(e,t,n=!1,r=!1){let s=m(e,"a","matMul"),a=m(t,"b","matMul");[s,a]=st(s,a);const o={a:s,b:a},i={transposeA:n,transposeB:r};return S.runKernel(yi,o,i)}const H=v({matMul_:Ag});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Og(e){const n={x:m(e,"x","sigmoid","float32")};return S.runKernel(gl,n)}const Qt=v({sigmoid_:Og});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Dg(e,t,n){const r=m(e,"x","slice","string_or_numeric");if(r.rank===0)throw new Error("Slicing scalar is not possible");const s={x:r},a={begin:t,size:n};return S.runKernel(pl,s,a)}const X=v({slice_:Dg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Fg(e){const n={x:m(e,"x","tanh","float32")};return S.runKernel(Pl,n)}const mr=v({tanh_:Fg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Rg(e,t,n,r,s,a){const o=m(e,"forgetBias","basicLSTMCell"),i=m(t,"lstmKernel","basicLSTMCell"),u=m(n,"lstmBias","basicLSTMCell"),l=m(r,"data","basicLSTMCell"),h=m(s,"c","basicLSTMCell"),c=m(a,"h","basicLSTMCell"),p=ht([l,c],1),d=H(p,i),g=z(d,u),N=g.shape[0],w=g.shape[1]/4,T=[N,w],x=X(g,[0,0],T),$=X(g,[0,w],T),E=X(g,[0,w*2],T),I=X(g,[0,w*3],T),A=z(B(Qt(x),mr($)),B(h,Qt(z(o,E)))),F=B(mr(A),Qt(I));return[A,F]}const Fc=v({basicLSTMCell_:Rg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Bg(e,t,n){const r=m(e,"x","batchToSpaceND"),s=t.reduce((i,u)=>i*u);y(r.rank>=1+t.length,()=>`input rank is ${r.rank} but should be > than blockShape.length ${t.length}`),y(n.length===t.length,()=>`crops.length is ${n.length} but should be equal to blockShape.length  ${t.length}`),y(r.shape[0]%s===0,()=>`input tensor batch is ${r.shape[0]} but is not divisible by the product of the elements of blockShape ${t.join(" * ")} === ${s}`);const a={x:r},o={blockShape:t,crops:n};return S.runKernel(bi,a,o)}const Qs=v({batchToSpaceND_:Bg});function Pg(e){let t;return e.rank===0||e.rank===1?t=O(e,[1,1,1,e.size]):e.rank===2?t=O(e,[1,1,e.shape[0],e.shape[1]]):e.rank===3?t=O(e,[1,e.shape[0],e.shape[1],e.shape[2]]):t=e,t}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Cg(e,t,n,r,s,a){a==null&&(a=.001);const o=m(e,"x","batchNorm"),i=m(t,"mean","batchNorm"),u=m(n,"variance","batchNorm");let l;s!=null&&(l=m(s,"scale","batchNorm"));let h;r!=null&&(h=m(r,"offset","batchNorm")),y(i.rank===u.rank,()=>"Batch normalization gradient requires mean and variance to have equal ranks."),y(h==null||i.rank===h.rank,()=>"Batch normalization gradient requires mean and offset to have equal ranks."),y(l==null||i.rank===l.rank,()=>"Batch normalization gradient requires mean and scale to have equal ranks.");const p={x:Pg(o),scale:l,offset:h,mean:i,variance:u},d={varianceEpsilon:a},g=S.runKernel(ru,p,d);return O(g,o.shape)}const Dn=v({batchNorm_:Cg});function Lg(e,t,n,r,s,a){const o=m(e,"x","batchNorm"),i=m(t,"mean","batchNorm"),u=m(n,"variance","batchNorm");let l;s!=null&&(l=m(s,"scale","batchNorm"));let h;return r!=null&&(h=m(r,"offset","batchNorm")),y(o.rank===2,()=>`Error in batchNorm2D: x must be rank 2 but got rank ${o.rank}.`),y(i.rank===2||i.rank===1,()=>`Error in batchNorm2D: mean must be rank 2 or rank 1 but got rank ${i.rank}.`),y(u.rank===2||u.rank===1,()=>`Error in batchNorm2D: variance must be rank 2 or rank 1 but got rank ${u.rank}.`),l!=null&&y(l.rank===2||l.rank===1,()=>`Error in batchNorm2D: scale must be rank 2 or rank 1 but got rank ${l.rank}.`),h!=null&&y(h.rank===2||h.rank===1,()=>`Error in batchNorm2D: offset must be rank 2 or rank 1 but got rank ${h.rank}.`),Dn(o,i,u,h,l,a)}const Rc=v({batchNorm2d_:Lg});function zg(e,t,n,r,s,a){const o=m(e,"x","batchNorm"),i=m(t,"mean","batchNorm"),u=m(n,"variance","batchNorm");let l;s!=null&&(l=m(s,"scale","batchNorm"));let h;return r!=null&&(h=m(r,"offset","batchNorm")),y(o.rank===3,()=>`Error in batchNorm3D: x must be rank 3 but got rank ${o.rank}.`),y(i.rank===3||i.rank===1,()=>`Error in batchNorm3D: mean must be rank 3 or rank 1 but got rank ${i.rank}.`),y(u.rank===3||u.rank===1,()=>`Error in batchNorm3D: variance must be rank 3 or rank 1 but got rank ${u.rank}.`),l!=null&&y(l.rank===3||l.rank===1,()=>`Error in batchNorm3D: scale must be rank 3 or rank 1 but got rank ${l.rank}.`),h!=null&&y(h.rank===3||h.rank===1,()=>`Error in batchNorm3D: offset must be rank 3 or rank 1 but got rank ${h.rank}.`),Dn(o,i,u,h,l,a)}const Bc=v({batchNorm3d_:zg});function Mg(e,t,n,r,s,a){const o=m(e,"x","batchNorm"),i=m(t,"mean","batchNorm"),u=m(n,"variance","batchNorm");let l;s!=null&&(l=m(s,"scale","batchNorm"));let h;return r!=null&&(h=m(r,"offset","batchNorm")),y(o.rank===4,()=>`Error in batchNorm4D: x must be rank 4 but got rank ${o.rank}.`),y(i.rank===4||i.rank===1,()=>`Error in batchNorm4D: mean must be rank 4 or rank 1 but got rank ${i.rank}.`),y(u.rank===4||u.rank===1,()=>`Error in batchNorm4D: variance must be rank 4 or rank 1 but got rank ${u.rank}.`),l!=null&&y(l.rank===4||l.rank===1,()=>`Error in batchNorm4D: scale must be rank 4 or rank 1 but got rank ${l.rank}.`),h!=null&&y(h.rank===4||h.rank===1,()=>`Error in batchNorm4D: offset must be rank 4 or rank 1 but got rank ${h.rank}.`),Dn(o,i,u,h,l,a)}const Pc=v({batchNorm4d_:Mg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vg(e,t,n){const r=m(e,"x","bincount"),s=m(t,"weights","bincount");y(r.dtype==="int32",()=>`Error in bincount: input dtype must be int32, but got ${r.dtype}`),y(n>=0,()=>`size must be non-negative, but got ${n}.`),y(s.size===r.size||s.size===0,()=>`Error in bincount: weights must have the same size as input or0-length, but got input shape: ${r.shape}, weights shape: ${s.shape}.`);const a={x:r,weights:s},o={size:n};return S.runKernel(wi,a,o)}const ta=v({bincount_:Vg});/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Wg(e,t){const n=m(e,"x","bitwiseAnd"),r=m(t,"y","bitwiseAnd");if(!Pt(n.shape,r.shape))throw new Error(`BitwiseAnd: Tensors must have the same shape. x: ${n.shape}, y: ${r.shape}`);if(n.dtype!=="int32"||r.dtype!=="int32")throw new Error(`BitwiseAnd: Only supports 'int32' values in tensor, found type of x: ${n.dtype} and type of y: ${r.dtype}`);const s={a:n,b:r};return S.runKernel(Ni,s)}const Cc=v({bitwiseAnd_:Wg});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ug(e,t){const n=m(e,"s0","broadcastArgs","int32"),r=m(t,"s1","broadcastArgs","int32");if(n.rank!==1)throw new Error(`broadcastArgs(): first input must be a vector (rank=1). Has rank ${n.rank}`);if(r.rank!==1)throw new Error(`broadcastArgs(): second input must be a vector (rank=1). Has rank ${r.rank}`);const s={s0:n,s1:r};return S.runKernel(vi,s)}const Lc=v({broadcastArgs_:Ug});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qg(e,t){let n=m(e,"broadcastTo","x");const r=n.shape;if($t(t),t.length<n.rank)throw new Error(`broadcastTo(): shape.length=${t.length} < input.rank=${n.rank}.`);if(t.length>n.rank){const l=n.shape.slice();for(;l.length<t.length;)l.unshift(1);n=O(n,l)}const s=n.shape,a=Array.from(t);for(let l=t.length-1;l>=0;l--)if(s[l]===t[l])a[l]=1;else if(n.shape[l]!==1)throw new Error(`broadcastTo(): [${r}] cannot be broadcast to [${t}].`);if(a.map((l,h)=>l>1?h:-1).filter(l=>l>=0).length===0)return Jt(n);const i={x:n},u={reps:a};return S.runKernel(zs,i,u)}const cn=v({broadcastTo_:qg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hg(e){const n={x:m(e,"x","ceil","float32")};return S.runKernel(Si,n)}const zc=v({ceil_:Hg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Je(e,t,n){$t(e),n=n||kn(t);const r={shape:e,value:t,dtype:n};return S.runKernel(Qi,{},r)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function jg(e,t,n){const r=m(e,"x","clipByValue");if(y(t<=n,()=>`Error in clip: min (${t}) must be less than or equal to max (${n}).`),t===n)return Je(r.shape,t,r.dtype);const s={x:r},a={clipValueMin:t,clipValueMax:n};return S.runKernel(Ti,s,a)}const Mc=v({clipByValue_:jg});function Gg(e){return ht(e,0)}const Vc=v({concat1d_:Gg});function Kg(e,t){return ht(e,t)}const Wc=v({concat2d_:Kg});function Xg(e,t){return ht(e,t)}const Uc=v({concat3d_:Xg});function Yg(e,t){return ht(e,t)}const qc=v({concat4d_:Yg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Zg(e,t,n,r,s="NHWC",a=[1,1],o){const i=m(e,"x","conv2d","float32"),u=m(t,"filter","conv2d","float32");let l=i,h=!1;i.rank===3&&(h=!0,l=O(i,[1,i.shape[0],i.shape[1],i.shape[2]])),y(l.rank===4,()=>`Error in conv2d: input must be rank 4, but got rank ${l.rank}.`),y(u.rank===4,()=>`Error in conv2d: filter must be rank 4, but got rank ${u.rank}.`),Ot("conv2d",r,o);const c=s==="NHWC"?l.shape[3]:l.shape[1];y(c===u.shape[2],()=>`Error in conv2d: depth of input (${c}) must match input depth for filter ${u.shape[2]}.`),y(ne(n,a),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`),y(Oe(a),()=>"Error in conv2D: Dilated rates should be larger than 0."),y(Oe(n),()=>"Error in conv2D: Strides should be larger than 0.");const p={x:l,filter:u},d={strides:n,pad:r,dataFormat:s,dilations:a,dimRoundingMode:o},g=S.runKernel(ki,p,d);return h?O(g,[g.shape[1],g.shape[2],g.shape[3]]):g}const Fn=v({conv2d_:Zg});function Jg(e,t,n,r,s="NWC",a=1,o){const i=m(e,"x","conv1d"),u=m(t,"filter","conv1d");let l=i,h=!1;i.rank===2&&(h=!0,l=O(i,[1,i.shape[0],i.shape[1]])),y(l.rank===3,()=>`Error in conv1d: input must be rank 3, but got rank ${l.rank}.`),y(u.rank===3,()=>`Error in conv1d: filter must be rank 3, but got rank ${u.rank}.`),Ot("conv1d",r,o),y(l.shape[2]===u.shape[1],()=>`Error in conv1d: depth of input (${l.shape[2]}) must match input depth for filter ${u.shape[1]}.`),y(ne(n,a),()=>`Error in conv1D: Either stride or dilation must be 1. Got stride ${n} and dilation '${a}'`),y(Oe(a),()=>"Error in conv1D: Dilated rates should be larger than 0."),y(Oe(n),()=>"Error in conv1D: Stride should be larger than 0."),y(s==="NWC",()=>`Error in conv1d: got dataFormat of ${s} but only NWC is currently supported.`);const c=O(u,[1,u.shape[0],u.shape[1],u.shape[2]]),p=O(l,[l.shape[0],1,l.shape[1],l.shape[2]]),w=Fn(p,c,[1,n],r,"NHWC",[1,a],o);return h?O(w,[w.shape[2],w.shape[3]]):O(w,[w.shape[0],w.shape[2],w.shape[3]])}const Hc=v({conv1d_:Jg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qg(e,t,n,r,s,a="NHWC",o){y(e.length===t.rank,()=>`Length of inShape (${e.length}) and rank of dy (${t.rank}) must match`);let i=e,u=t,l=!1;t.rank===3&&(l=!0,u=O(t,[1,t.shape[0],t.shape[1],t.shape[2]]),i=[1,e[0],e[1],e[2]]),y(i.length===4,()=>`Error in conv2dDerInput: inShape must be length 4, but got length ${i.length}.`),y(u.rank===4,()=>`Error in conv2dDerInput: dy must be rank 4, but got rank ${u.rank}`),y(n.rank===4,()=>`Error in conv2dDerInput: filter must be rank 4, but got rank ${n.rank}`);const h=a==="NHWC"?i[3]:i[1],c=a==="NHWC"?u.shape[3]:u.shape[1];y(h===n.shape[2],()=>`Error in conv2dDerInput: depth of input (${h}) must match input depth for filter ${n.shape[2]}.`),y(c===n.shape[3],()=>`Error in conv2dDerInput: depth of output (${c}) must match output depth for filter ${n.shape[3]}.`),Ot("conv2dDerInput",s,o);const p={dy:u,filter:n},d={strides:r,pad:s,dataFormat:a,dimRoundingMode:o,inputShape:i},g=S.runKernel(xi,p,d);return l?O(g,[g.shape[1],g.shape[2],g.shape[3]]):g}const jc=v({conv2DBackpropInput_:Qg});function ty(e,t,n,r,s,a){const o=m(e,"x","conv2dTranspose"),i=m(t,"filter","conv2dTranspose");return jc(n,o,i,r,s,"NHWC",a)}const Gc=v({conv2dTranspose_:ty});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ey(e,t,n,r,s="NDHWC",a=[1,1,1]){const o=m(e,"x","conv3d"),i=m(t,"filter","conv3d");let u=o,l=!1;o.rank===4&&(l=!0,u=O(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]])),y(u.rank===5,()=>`Error in conv3d: input must be rank 5, but got rank ${u.rank}.`),y(i.rank===5,()=>`Error in conv3d: filter must be rank 5, but got rank ${i.rank}.`),y(u.shape[4]===i.shape[3],()=>`Error in conv3d: depth of input (${u.shape[4]}) must match input depth for filter ${i.shape[3]}.`),y(ne(n,a),()=>`Error in conv3D: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`),y(s==="NDHWC",()=>`Error in conv3d: got dataFormat of ${s} but only NDHWC is currently supported.`),y(Oe(a),()=>"Error in conv3D: Dilated rates should be larger than 0."),y(Oe(n),()=>"Error in conv3D: Strides should be larger than 0.");const h={x:u,filter:i},c={strides:n,pad:r,dataFormat:s,dilations:a},p=S.runKernel(Ai,h,c);return l?O(p,[p.shape[1],p.shape[2],p.shape[3],p.shape[4]]):p}const Kc=v({conv3d_:ey});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ny(e,t,n,r,s){y(e.length===t.rank,()=>`Length of inShape (${e.length}) and rank of dy (${t.rank}) must match`);let a=e,o=t,i=!1;t.rank===4&&(i=!0,o=O(t,[1,t.shape[0],t.shape[1],t.shape[2],t.shape[3]]),a=[1,e[0],e[1],e[2],e[3]]);const u=a[4],l=o.shape[4];y(a.length===5,()=>`Error in conv3dDerInput: inShape must be length 5, but got length ${a.length}.`),y(o.rank===5,()=>`Error in conv3dDerInput: dy must be rank 5, but got rank ${o.rank}`),y(n.rank===5,()=>`Error in conv3dDerInput: filter must be rank 5, but got rank ${n.rank}`),y(u===n.shape[3],()=>`Error in conv3dDerInput: depth of input (${u}) must match input depth for filter ${n.shape[3]}.`),y(l===n.shape[4],()=>`Error in conv3dDerInput: depth of output (${l}) must match output depth for filter ${n.shape[4]}.`);const h={dy:o,filter:n},c={pad:s,strides:r,inputShape:a},p=S.runKernel(Oi,h,c);return i?O(p,[p.shape[1],p.shape[2],p.shape[3],p.shape[4]]):p}const ry=v({conv3DBackpropInput_:ny});function sy(e,t,n,r,s){const a=m(e,"x","conv3dTranspose"),o=m(t,"filter","conv3dTranspose");return ry(n,a,o,r,s)}const Xc=v({conv3dTranspose_:sy});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ay(e){const n={x:m(e,"x","cos","float32")};return S.runKernel(Di,n)}const Yc=v({cos_:ay});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function oy(e){const n={x:m(e,"x","cosh","float32")};return S.runKernel(Fi,n)}const Zc=v({cosh_:oy});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function iy(e,t=0,n=!1,r=!1){const a={x:m(e,"x","cumprod")},o={axis:t,exclusive:n,reverse:r};return S.runKernel(Ri,a,o)}const Jc=v({cumprod_:iy});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function uy(e,t=0,n=!1,r=!1){const a={x:m(e,"x","cumsum")},o={axis:t,exclusive:n,reverse:r};return S.runKernel(Bi,a,o)}const Qc=v({cumsum_:uy});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ly(e,t,n,r=!1){const s=m(e,"x","denseBincount"),a=m(t,"weights","denseBincount");y(s.dtype==="int32",()=>`Error in denseBincount: input dtype must be int32, but got ${s.dtype}`),y(s.rank<=2,()=>`Error in denseBincount: input must be at most rank 2, but got rank ${s.rank}.`),y(n>=0,()=>`size must be non-negative, but got ${n}.`),y(a.size===s.size||a.size===0,()=>`Error in denseBincount: weights must have the same shape as x or 0-length, but got x shape: ${s.shape}, weights shape: ${a.shape}.`);const o={x:s,weights:a},i={size:n,binaryOutput:r};return S.runKernel(Ci,o,i)}const th=v({denseBincount_:ly});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function cy(e,t,n="NHWC"){const r=m(e,"x","depthToSpace","float32"),s=n==="NHWC"?r.shape[1]:r.shape[2],a=n==="NHWC"?r.shape[2]:r.shape[3],o=n==="NHWC"?r.shape[3]:r.shape[1];y(t>1,()=>`blockSize should be > 1 for depthToSpace, but was: ${t}`),y(s*t>=0,()=>`Negative dimension size caused by overflow when multiplying
    ${s} and ${t}  for depthToSpace with input shape
    ${r.shape}`),y(a*t>=0,()=>`Negative dimension size caused by overflow when multiplying
    ${a} and ${t} for depthToSpace with input shape
        ${r.shape}`),y(o%(t*t)===0,()=>`Dimension size must be evenly divisible by ${t*t} but is ${o} for depthToSpace with input shape ${r.shape}`);const i={x:r},u={blockSize:t,dataFormat:n};return S.runKernel(Li,i,u)}const eh=v({depthToSpace_:cy});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hy(e,t,n,r,s="NHWC",a=[1,1],o){const i=m(e,"x","depthwiseConv2d","float32"),u=m(t,"filter","depthwiseConv2d","float32");let l=i,h=!1;i.rank===3&&(h=!0,l=O(i,[1,i.shape[0],i.shape[1],i.shape[2]])),y(l.rank===4,()=>`Error in depthwiseConv2d: input must be rank 4, but got rank ${l.rank}.`),y(u.rank===4,()=>`Error in depthwiseConv2d: filter must be rank 4, but got rank ${u.rank}.`);const c=s==="NHWC"?l.shape[3]:l.shape[1];y(c===u.shape[2],()=>`Error in depthwiseConv2d: number of input channels (${c}) must match the inChannels dimension in filter ${u.shape[2]}.`),Ot("depthwiseConv2d",r,o);const p={x:l,filter:u},d={strides:n,pad:r,dataFormat:s,dilations:a,dimRoundingMode:o},g=S.runKernel(zi,p,d);return h?O(g,[g.shape[1],g.shape[2],g.shape[3]]):g}const Er=v({depthwiseConv2d_:hy});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function py(e){const n={x:m(e,"x","diag")};return S.runKernel(Wi,n)}const nh=v({diag_:py});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fy(e,t,n,r,s=[1,1],a="NHWC"){const o=m(e,"x","dilation2d"),i=m(t,"filter","dilation2d");y(o.rank===3||o.rank===4,()=>`Error in dilation2d: input must be rank 3 or 4, but got rank ${o.rank}.`),y(i.rank===3,()=>`Error in dilation2d: filter must be rank 3, but got rank ${i.rank}.`),y(a==="NHWC",()=>`Error in dilation2d: Only NHWC is currently supported, but got dataFormat of ${a}`);let u=o,l=!1;o.rank===3&&(u=O(o,[1,o.shape[0],o.shape[1],o.shape[2]]),l=!0),y(u.shape[3]===i.shape[2],()=>`Error in dilation2d:  input and filter must have the same depth: ${u.shape[3]} vs ${i.shape[2]}`);const h={x:u,filter:i},c={strides:n,pad:r,dilations:s},p=S.runKernel(Ui,h,c);return l?O(p,[p.shape[1],p.shape[2],p.shape[3]]):p}const rh=v({dilation2d_:fy});/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function sh(e,t){const n=e.length,r=[];for(let s=0;s<n;s++){const a=n-1-s,o=e[a]||1;(t[t.length-1-s]||1)>1&&o===1&&r.unshift(a)}return r}function ea(e,t){const n=[];for(let r=0;r<t.length;r++){const s=e[e.length-r-1],a=t.length-r-1,o=t[a];(s==null||s===1&&o>1)&&n.unshift(a)}return n}function at(e,t){const n=Math.max(e.length,t.length),r=new Array(n);for(let s=0;s<n;s++){let a=e[e.length-s-1];a==null&&(a=1);let o=t[t.length-s-1];if(o==null&&(o=1),a===1)r[n-s-1]=o;else if(o===1)r[n-s-1]=a;else if(a!==o){const i=`Operands could not be broadcast together with shapes ${e} and ${t}.`;throw Error(i)}else r[n-s-1]=a}return r}const dy=Object.freeze(Object.defineProperty({__proto__:null,assertAndGetBroadcastShape:at,getBroadcastDims:sh,getReductionAxes:ea},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function my(e,t){let n=m(e,"a","equal","string_or_numeric"),r=m(t,"b","equal","string_or_numeric");[n,r]=st(n,r),at(n.shape,r.shape);const s={a:n,b:r};return S.runKernel(Ki,s)}const Rn=v({equal_:my});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function gy(e,t,n){const r=m(t,"a","where"),s=m(n,"b","where"),a=m(e,"condition","where","bool"),o=at(at(a.shape,r.shape),s.shape),i=cn(a,o),u=cn(r,o),l=cn(s,o),h={condition:i,t:u,e:l};return S.runKernel(cl,h)}const te=v({where_:gy});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function yy(e){const n={x:m(e,"x","zerosLike")};return S.runKernel(Wl,n)}const Et=v({zerosLike_:yy});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function by(e,t){let n=m(e,"a","div"),r=m(t,"b","div");[n,r]=st(n,r);const s=Y(n,r),a=Et(s),o=Rn(r,a);return te(o,a,s)}const ah=v({divNoNan_:by});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function wy(e,t){const n=m(e,"t1","dot"),r=m(t,"t2","dot");y((n.rank===1||n.rank===2)&&(r.rank===1||r.rank===2),()=>`Error in dot: inputs must all be rank 1 or 2, but got ranks ${n.rank} and ${r.rank}.`);const s=n.rank===1?n.size:n.shape[1],a=r.rank===1?r.size:r.shape[0];if(y(s===a,()=>`Error in dot: inner dimensions of inputs must match, but got ${s} and ${a}.`),n.rank===1&&r.rank===1){const o=O(n,[1,-1]),i=O(r,[-1,1]),u=H(o,i);return O(u,[])}else if(n.rank===1&&r.rank===2){const o=O(n,[1,-1]),i=O(r,[r.shape[0],r.shape[1]]),u=H(o,i);return O(u,[u.size])}else if(n.rank===2&&r.rank===1){const o=O(r,[-1,1]),i=H(n,o);return O(i,[i.size])}else{const o=O(r,[r.shape[0],r.shape[1]]);return H(n,o)}}const oh=v({dot_:wy});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ny(e,...t){const n=t.map((s,a)=>m(s,`tensors${a}`,"einsum")),r={equation:e};return S.runKernel(Hi,n,r)}const Te=v({einsum_:Ny});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vy(e){const n={x:m(e,"x","elu","float32")};return S.runKernel(ji,n)}const na=v({elu_:vy});/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Sy(e,t){const n=m(e,"x","ensureShape","string_or_numeric");if(!Ho(n.shape,t))throw new Error(`EnsureShape: Shape of tensor ${n.shape} is not compatible with expected shape ${t}`);return e}const ih=v({ensureShape_:Sy});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ty(e){let t=m(e,"x","erf");y(t.dtype==="int32"||t.dtype==="float32",()=>"Input dtype must be `int32` or `float32`."),t.dtype==="int32"&&(t=j(t,"float32"));const n={x:t};return S.runKernel(Gi,n)}const uh=v({erf_:Ty});/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ra(e,t){for(let n=0;n<e.length;++n)if(e[e.length-n-1]!==t-1-n)return!1;return!0}function lh(e,t,n){const r=e.length+t.length,s=[];let a=0,o=0;for(let i=0;i<r;i++)n.indexOf(i)===-1?s.push(e[a++]):s.push(t[o++]);return s}function Ey(e,t){const n=[],r=e.length;for(let a=0;a<r;a++)t.indexOf(a)===-1&&n.push(e[a]);const s=t.map(a=>e[a]);return[n,s]}function Bn(e,t){const n=t.map(r=>1);return lh(e,n,t)}function $y(e,t,n){y(ra(t,n),()=>`${e} supports only inner-most axes for now. Got axes ${t} and rank-${n} input.`)}function _y(e,t){if(ra(e,t))return null;const n=[];for(let r=0;r<t;++r)e.indexOf(r)===-1&&n.push(r);return e.forEach(r=>n.push(r)),n}function ky(e){return e.map((t,n)=>[n,t]).sort((t,n)=>t[1]-n[1]).map(t=>t[0])}function Iy(e,t){const n=[];for(let r=t-e;r<t;++r)n.push(r);return n}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xy(e,t=null,n=!1){const s={x:m(e,"x","max")},a={reductionIndices:t,keepDims:n};return S.runKernel(Tu,s,a)}const ke=v({max_:xy});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ay(e,t=null,n=!1){const s={x:m(e,"x","min")},a={axis:t,keepDims:n};return S.runKernel(xu,s,a)}const gr=v({min_:Ay});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Oy(e,t){let n=m(e,"base","pow"),r=m(t,"exp","pow");[n,r]=st(n,r);const s={a:n,b:r};return S.runKernel(qu,s)}const Ge=v({pow_:Oy});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function U(e,t){if((ut(e)&&t!=="string"||Array.isArray(e))&&t!=="complex64")throw new Error("Error creating a new Scalar: value must be a primitive (number|boolean|string)");if(t==="string"&&ut(e)&&!(e instanceof Uint8Array))throw new Error("When making a scalar from encoded string, the value must be `Uint8Array`.");return ye(e,[],[],t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Dy(e){const n={x:m(e,"x","sqrt","float32")};return S.runKernel(bl,n)}const Ht=v({sqrt_:Dy});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Fy(e){const t=m(e,"x","square"),n={};return S.runKernel("Square",{x:t},n)}const At=v({square_:Fy});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ry(e,t=null,n=!1){let r=m(e,"x","sum");r.dtype==="bool"&&(r=j(r,"int32"));const s={x:r},a={axis:t,keepDims:n};return S.runKernel(wl,s,a)}const Q=v({sum_:Ry});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function By(e,t="euclidean",n=null,r=!1){e=m(e,"x","norm");const s=ch(e,t,n);let a=s.shape;if(r){const o=_n(n,e.shape);a=Bn(s.shape,o)}return O(s,a)}function ch(e,t,n=null){if(e.rank===0)return Tt(e);if(e.rank!==1&&n===null)return ch(O(e,[-1]),t,n);if(e.rank===1||typeof n=="number"||Array.isArray(n)&&n.length===1){if(t===1)return Q(Tt(e),n);if(t===1/0)return ke(Tt(e),n);if(t===-1/0)return gr(Tt(e),n);if(t==="euclidean"||t===2)return Ht(Q(Ge(Tt(e),U(2,"int32")),n));throw new Error(`Error in norm: invalid ord value: ${t}`)}if(Array.isArray(n)&&n.length===2){if(t===1)return ke(Q(Tt(e),n[0]),n[1]-1);if(t===1/0)return ke(Q(Tt(e),n[1]),n[0]);if(t===-1/0)return gr(Q(Tt(e),n[1]),n[0]);if(t==="fro"||t==="euclidean")return Ht(Q(At(e),n));throw new Error(`Error in norm: invalid ord value: ${t}`)}throw new Error(`Error in norm: invalid axis: ${n}`)}const Pn=v({norm_:By});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Py(e,t=null,n=!1){return Pn(e,"euclidean",t,n)}const hh=v({euclideanNorm_:Py});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Cy(e){const n={x:m(e,"x","exp")};return S.runKernel(Xi,n)}const de=v({exp_:Cy});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ly(e,t=0){const n=m(e,"x","expandDims","string_or_numeric");y(t<=n.rank,()=>"Axis must be <= rank of the tensor");const r={input:n},s={dim:t};return S.runKernel(Yi,r,s)}const _t=v({expandDims_:Ly});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zy(e){const n={x:m(e,"x","expm1")};return S.runKernel(Zi,n)}const ph=v({expm1_:zy});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function My(e,t){const n=m(e,"x","tile","string_or_numeric");y(n.rank===t.length,()=>`Error in transpose: rank of input ${n.rank} must match length of reps ${t}.`);const r={x:n},s={reps:t};return S.runKernel(zs,r,s)}const We=v({tile_:My});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vy(e,t,n,r="float32"){t==null&&(t=e);const s=qt([e,t],r),a=e<=t?e:t;for(let i=0;i<a;++i)s.set(1,i,i);const o=O(s.toTensor(),[e,t]);if(n==null)return o;if(n.length===1)return We(_t(o,0),[n[0],1,1]);if(n.length===2)return We(_t(_t(o,0),0),[n[0],n[1],1,1]);if(n.length===3)return We(_t(_t(_t(o,0),0),0),[n[0],n[1],n[2],1,1]);throw new Error(`eye() currently supports only 1D and 2D batchShapes, but received ${n.length}D.`)}const sa=v({eye_:Vy});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Wy(e){const n={x:m(e,"x","floor","float32")};return S.runKernel(eu,n)}const aa=v({floor_:Wy});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Uy(e,t,n=0,r=0){const s=m(e,"x","gather"),a=m(t,"indices","gather","int32"),o={x:s,indices:a},i={axis:n,batchDims:r};return S.runKernel(su,o,i)}const oa=v({gather_:Uy});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qy(e,t){let n=m(e,"a","greater","string_or_numeric"),r=m(t,"b","greater","string_or_numeric");[n,r]=st(n,r),at(n.shape,r.shape);const s={a:n,b:r};return S.runKernel(ou,s)}const Qe=v({greater_:qy});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hy(e,t){let n=m(e,"a","greaterEqual","string_or_numeric"),r=m(t,"b","greaterEqual","string_or_numeric");[n,r]=st(n,r),at(n.shape,r.shape);const s={a:n,b:r};return S.runKernel(iu,s)}const ia=v({greaterEqual_:Hy});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function jy(e){const n={input:m(e,"input","imag")};return S.runKernel(lu,n)}const Cn=v({imag_:jy});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Gy(e){const n={x:m(e,"x","isFinite")};return S.runKernel(cu,n)}const fh=v({isFinite_:Gy});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ky(e){const n={x:m(e,"x","isInf")};return S.runKernel(hu,n)}const dh=v({isInf_:Ky});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xy(e){const n={x:m(e,"x","isNaN")};return S.runKernel(pu,n)}const mh=v({isNaN_:Xy});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Yy(e,t=.2){const r={x:m(e,"x","leakyRelu")},s={alpha:t};return S.runKernel(fu,r,s)}const ua=v({leakyRelu_:Yy});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Zy(e,t){let n=m(e,"a","less","string_or_numeric"),r=m(t,"b","less","string_or_numeric");[n,r]=st(n,r),at(n.shape,r.shape);const s={a:n,b:r};return S.runKernel(du,s)}const yr=v({less_:Zy});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Jy(e,t){let n=m(e,"a","lessEqual","string_or_numeric"),r=m(t,"b","lessEqual","string_or_numeric");[n,r]=st(n,r),at(n.shape,r.shape);const s={a:n,b:r};return S.runKernel(mu,s)}const $r=v({lessEqual_:Jy});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function gh(e,t,n){if(n<=0)throw new Error("The number of values should be positive.");const r={start:e,stop:t,num:n};return S.runKernel(gu,{},r)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qy(e,t=5,n=1,r=1,s=.5){const a=m(e,"x","localResponseNormalization");y(a.rank===4||a.rank===3,()=>`Error in localResponseNormalization: x must be rank 3 or 4 but got
               rank ${a.rank}.`),y(qe(t),()=>`Error in localResponseNormalization: depthRadius must be an integer but got depthRadius ${t}.`);let o=a,i=!1;a.rank===3&&(i=!0,o=O(a,[1,a.shape[0],a.shape[1],a.shape[2]]));const u={x:o},l={depthRadius:t,bias:n,alpha:r,beta:s},h=S.runKernel(Su,u,l);return i?O(h,[h.shape[1],h.shape[2],h.shape[3]]):h}const yh=v({localResponseNormalization_:Qy});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function tb(e){const n={x:m(e,"x","log","float32")};return S.runKernel(yu,n)}const Ke=v({log_:tb});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function eb(e){const n={x:m(e,"x","log1p")};return S.runKernel(bu,n)}const la=v({log1p_:eb});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function nb(e){return y(ce(e),()=>"The f passed in grad(f) must be a function"),(t,n)=>{const r=m(t,"x","tf.grad","string_or_numeric"),s=n!=null?m(n,"dy","tf.grad"):null;return S.tidy(()=>{const{value:a,grads:o}=S.gradients(()=>e(r),[r],s);return s!=null&&gt(a.shape,s.shape,"The shape of dy passed in grad(f)(x, dy) must match the shape returned by f(x)"),_r(o),o[0]})}}function rb(e){return y(ce(e),()=>"The f passed in grads(f) must be a function"),(t,n)=>{y(Array.isArray(t),()=>"The args passed in grads(f)(args) must be an array of `Tensor`s or `TensorLike`s");const r=gn(t,"args","tf.grads","string_or_numeric"),s=n!=null?m(n,"dy","tf.grads"):null;return S.tidy(()=>{const{value:a,grads:o}=S.gradients(()=>e(...r),r,s);return s!=null&&gt(a.shape,s.shape,"The shape of dy passed in grads(f)([x1,...], dy) must match the shape returned by f([x1,...])"),_r(o),o})}}function sb(e){return y(ce(e),()=>"The f passed in valueAndGrad(f) must be a function"),(t,n)=>{y(t instanceof et,()=>"The x passed in valueAndGrad(f)(x) must be a tensor"),y(n==null||n instanceof et,()=>"The dy passed in valueAndGrad(f)(x, dy) must be a tensor");const{grads:r,value:s}=S.gradients(()=>e(t),[t],n);return _r(r),{grad:r[0],value:s}}}function ab(e){return y(ce(e),()=>"The f passed in valueAndGrads(f) must be a function"),(t,n)=>{y(Array.isArray(t)&&t.every(s=>s instanceof et),()=>"The args passed in valueAndGrads(f)(args) must be array of tensors"),y(n==null||n instanceof et,()=>"The dy passed in valueAndGrads(f)(args, dy) must be a tensor");const r=S.gradients(()=>e(...t),t,n);return n!=null&&gt(r.value.shape,n.shape,"The shape of dy passed in valueAndGrads(f)([x1,...], dy) must match the shape returned by f([x1,...])"),_r(r.grads),r}}function bh(e,t){y(ce(e),()=>"The f passed in variableGrads(f) must be a function"),y(t==null||Array.isArray(t)&&t.every(l=>l instanceof mn),()=>"The varList passed in variableGrads(f, varList) must be an array of variables");const n=t!=null;if(!n){t=[];for(const l in S.registeredVariables)t.push(S.registeredVariables[l])}const r=n?t.filter(l=>!l.trainable):null,s=t.length;t=t.filter(l=>l.trainable),y(t.length>0,()=>`variableGrads() expects at least one of the input variables to be trainable, but none of the ${s} variables is trainable.`);const a=!0,{value:o,grads:i}=S.gradients(e,t,null,a);y(i.some(l=>l!=null),()=>"Cannot find a connection between any variable and the result of the loss function y=f(x). Please make sure the operations that use variables are inside the function f passed to minimize()."),y(o.rank===0,()=>`The f passed in variableGrads(f) must return a scalar, but it returned a rank-${o.rank} tensor`);const u={};return t.forEach((l,h)=>{i[h]!=null&&(u[l.name]=i[h])}),r!=null&&r.forEach(l=>u[l.name]=null),{value:o,grads:u}}function jt(e){return S.customGrad(e)}function _r(e){if(e.filter(n=>n==null).length>0)throw new Error(`Cannot compute gradient of y=f(x) with respect to x. Make sure that
    the f you passed encloses all operations that lead from x to y.`)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ob(e){const n={x:m(e,"x","neg")};return S.runKernel(Bu,n)}const Bt=v({neg_:ob});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ib(e){const n={x:m(e,"x","softplus")};return S.runKernel(yl,n)}const ca=v({softplus_:ib});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ub(e){const t=m(e,"x","logSigmoid");return jt(r=>({value:Bt(ca(Bt(r))),gradFunc:o=>B(o,Qt(Bt(r)))}))(t)}const wh=v({logSigmoid_:ub});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function lb(e,t){let n=m(e,"a","sub"),r=m(t,"b","sub");[n,r]=st(n,r);const s={a:n,b:r};return S.runKernel(Rl,s)}const W=v({sub_:lb});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function cb(e,t=-1){const n=m(e,"logits","logSoftmax");if(t===-1&&(t=n.rank-1),t!==n.rank-1)throw Error(`Log Softmax along a non-last dimension is not yet supported. Logits was rank ${n.rank} and axis was ${t}`);return jt((s,a)=>{const i=ke(s,t,!0),u=W(s,i),l=W(j(u,"float32"),Ke(Q(de(u),t,!0)));return a([l]),{value:l,gradFunc:(c,p)=>{const[d]=p,g=!0,N=de(d);return W(c,B(Q(c,t,g),N))}}})(n)}const Nh=v({logSoftmax_:cb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hb(e,t=null,n=!1){const r=m(e,"x","logSumExp"),s=_n(t,r.shape),a=ke(r,s,!0),o=W(r,a),i=de(o),u=Q(i,s),l=Ke(u),h=z(O(a,l.shape),l);if(n){const c=Bn(h.shape,s);return O(h,c)}return h}const ha=v({logSumExp_:hb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function pb(e,t){const n=m(e,"a","logicalAnd","bool"),r=m(t,"b","logicalAnd","bool");at(n.shape,r.shape);const s={a:n,b:r};return S.runKernel(wu,s)}const Nn=v({logicalAnd_:pb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fb(e){const n={x:m(e,"x","logicalNot","bool")};return S.runKernel(Nu,n)}const pa=v({logicalNot_:fb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function db(e,t){const n=m(e,"a","logicalOr","bool"),r=m(t,"b","logicalOr","bool");at(n.shape,r.shape);const s={a:n,b:r};return S.runKernel(vu,s)}const fa=v({logicalOr_:db});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function mb(e,t){const n=m(e,"a","logicalXor","bool"),r=m(t,"b","logicalXor","bool");return at(n.shape,r.shape),Nn(fa(e,t),pa(Nn(e,t)))}const vh=v({logicalXor_:mb});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Wn=2147483648;function gb(e,t,n="left"){const r=m(e,"sortedSequence","searchSorted"),s=m(t,"values","searchSorted"),a=r.shape[r.shape.length-1],o=s.shape[s.shape.length-1],i=O(r,[-1,a]),u=O(s,[-1,o]);if(i.rank<2)throw new Error("Sorted input argument must be at least 2-dimensional");if(i.shape[0]!==u.shape[0])throw new Error("Leading dimension of 'sortedSequence' and 'values' must match.");if(K(u.shape)>=Wn)throw new Error(`values tensor size must less than ${Wn}`);if(i.shape[1]>=Wn)throw new Error(`trailing dim_size must less than ${Wn} for int32 output type, was ${i.shape[1]}`);const l={sortedSequence:i,values:u},h={side:n};return S.runKernel(ll,l,h)}const kr=v({searchSorted_:gb});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Sh(e,t){return kr(e,t,"left")}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function yb(e,t,n,r,s){const a=m(e,"x","maxPool"),o=1;let i=a,u=!1;a.rank===3&&(u=!0,i=O(a,[1,a.shape[0],a.shape[1],a.shape[2]])),y(i.rank===4,()=>`Error in maxPool: input must be rank 4 but got rank ${i.rank}.`),y(ne(n,o),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${n} and dilations '${o}'`),Ot("maxPool",r,s);const l={x:i},h={filterSize:t,strides:n,pad:r,dimRoundingMode:s},c=S.runKernel($u,l,h);return u?O(c,[c.shape[1],c.shape[2],c.shape[3]]):c}const da=v({maxPool_:yb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function bb(e,t=[1,1,1],n,r,s,a="NDHWC"){const o=m(e,"x","maxPool3d");let i=o,u=!1;o.rank===4&&(u=!0,i=O(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]])),y(i.rank===5,()=>`Error in maxPool3d: x must be rank 5 but got rank ${i.rank}.`),y(a==="NDHWC",()=>`Error in maxPool3d: Only NDHWC is currently supported, but got dataFormat of ${a}`),Ot("maxPool3d",r,s);const l={x:i},h={filterSize:t,strides:n,pad:r,dimRoundingMode:s,dataFormat:a},c=S.runKernel(_u,l,h);return u?O(c,[c.shape[1],c.shape[2],c.shape[3],c.shape[4]]):c}const Th=v({maxPool3d_:bb});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function wb(e,t,n,r,s=!1){const o={x:m(e,"x","maxPoolWithArgmax")},i={filterSize:t,strides:n,pad:r,includeBatchInIndex:s},u=S.runKernel(ku,o,i);return{result:u[0],indexes:u[1]}}const Eh=v({maxPoolWithArgmax_:wb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Nb(e,t){let n=m(e,"a","maximum"),r=m(t,"b","maximum");[n,r]=st(n,r),n.dtype==="bool"&&(n=j(n,"int32"),r=j(r,"int32")),at(n.shape,r.shape);const s={a:n,b:r};return S.runKernel(Eu,s)}const ma=v({maximum_:Nb});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vb(e,t=null,n=!1){const s={x:m(e,"x","mean")},a={axis:t,keepDims:n};return S.runKernel(Iu,s,a)}const vn=v({mean_:vb});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function De(e,t="float32"){if($t(e),t==="complex64"){const r=De(e,"float32"),s=De(e,"float32");return ee(r,s)}const n=vr(K(e),t);return S.makeTensor(n,e,t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ue(e,t="float32"){if($t(e),t==="complex64"){const r=ue(e,"float32"),s=De(e,"float32");return ee(r,s)}const n=Ds(K(e),t);return S.makeTensor(n,e,t)}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $h(e,t,{indexing:n="xy"}={}){if(n!=="xy"&&n!=="ij")throw new TypeError(`${n} is not a valid third argument to meshgrid`);if(e===void 0)return[];let r=m(e,"x","meshgrid",e instanceof et?e.dtype:"float32");if(t===void 0)return[r];let s=m(t,"y","meshgrid",t instanceof et?t.dtype:"float32");const a=K(r.shape),o=K(s.shape);return n==="xy"?(r=O(r,[1,-1]),s=O(s,[-1,1]),[H(ue([o,1],r.dtype),r),H(s,ue([1,a],s.dtype))]):(r=O(r,[-1,1]),s=O(s,[1,-1]),[H(r,ue([1,o],r.dtype)),H(ue([a,1],s.dtype),s)])}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Sb(e,t){let n=m(e,"a","minimum"),r=m(t,"b","minimum");[n,r]=st(n,r),n.dtype==="bool"&&(n=j(n,"int32"),r=j(r,"int32")),at(n.shape,r.shape);const s={a:n,b:r};return S.runKernel(Au,s)}const Sn=v({minimum_:Sb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Tb(e,t,n){y(n==="reflect"||n==="symmetric",()=>`Invalid mode. Mode must be either reflect or symmetric. Got ${n}.`);const r=m(e,"x","mirrorPad");if(r.rank===0)throw new Error("mirrorPad(scalar) is not defined. Pass non-scalar to mirrorPad");y(t.length===r.rank,()=>`Padding doesn't match input. Must be ${r.rank}. Got ${t.length}.`);const s=n==="reflect"?1:0;for(let i=0;i<r.rank;i++)y(t[i].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),y(t[i][0]>=0&&t[i][0]<=r.shape[i]-s&&t[i][1]>=0&&t[i][1]<=r.shape[i]-s,()=>`Padding in dimension ${i} cannot be greater than or equal to ${r.shape[i]-s} or less than 0 for input of shape ${r.shape}`);const a={paddings:t,mode:n},o={x:r};return S.runKernel(Ou,o,a)}const _h=v({mirrorPad_:Tb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Eb(e,t){let n=m(e,"a","mod"),r=m(t,"b","mod");[n,r]=st(n,r);const s={a:n,b:r};return S.runKernel(Du,s)}const kh=v({mod_:Eb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $b(e,t=null,n=!1){e=m(e,"x","moments");const r=_n(t,e.shape),s=vn(e,r,n);let a=s.shape;n||(a=Bn(s.shape,r));const o=At(W(j(e,"float32"),O(s,a))),i=vn(o,r,n);return{mean:s,variance:i}}const Ih=v({moments_:$b});function _b(e,t,n,r){const s=m(t,"data","multiRNNCell"),a=gn(n,"c","multiRNNCell"),o=gn(r,"h","multiRNNCell");let i=s;const u=[];for(let c=0;c<e.length;c++){const p=e[c](i,a[c],o[c]);u.push(p[0]),u.push(p[1]),i=p[1]}const l=[],h=[];for(let c=0;c<u.length;c+=2)l.push(u[c]),h.push(u[c+1]);return[l,h]}const xh=v({multiRNNCell_:_b});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function kb(e,t,n,r=!1){const s=m(e,"logits","multinomial"),a=s.size,o=s.rank;if(a<2)throw new Error(`Error in multinomial: you need at least 2 outcomes, but got ${a}.`);if(o>2)throw new Error(`Rank of probabilities must be 1 or 2, but is ${o}`);n=n||Math.random();const u={logits:o===1?O(s,[1,-1]):s},l={numSamples:t,seed:n,normalized:r},h=S.runKernel(Fu,u,l);return o===1?O(h,[h.size]):h}const Ah=v({multinomial_:kb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ib(e,t){let n=m(e,"a","notEqual","string_or_numeric"),r=m(t,"b","notEqual","string_or_numeric");[n,r]=st(n,r),at(n.shape,r.shape);const s={a:n,b:r};return S.runKernel(Pu,s)}const ga=v({notEqual_:Ib});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xb(e,t,n=1,r=0,s="int32"){if(t<2)throw new Error(`Error in oneHot: depth must be >=2, but it is ${t}`);const o={indices:m(e,"indices","oneHot","int32")},i={dtype:s,depth:t,onValue:n,offValue:r};return S.runKernel(Vu,o,i)}const Tn=v({oneHot_:xb});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ab(e){const n={x:m(e,"x","onesLike")};return S.runKernel(Mu,n)}const Oh=v({onesLike_:Ab});function Ob(e,t){const n=m(e,"v1","outerProduct"),r=m(t,"v2","outerProduct");y(n.rank===1&&r.rank===1,()=>`Error in outerProduct: inputs must be rank 1, but got ranks ${n.rank} and ${r.rank}.`);const s=O(n,[-1,1]),a=O(r,[1,-1]);return H(s,a)}const Dh=v({outerProduct_:Ob});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Db(e,t,n=0){const r=m(e,"x","pad");if(r.rank===0)throw new Error("pad(scalar) is not defined. Pass non-scalar to pad");const s={paddings:t,constantValue:n},a={x:r};return S.runKernel(Uu,a,s)}const tn=v({pad_:Db});function Fb(e,t,n=0){return y(t.length===2,()=>"Invalid number of paddings. Must be length of 2."),tn(e,[t],n)}const Fh=v({pad1d_:Fb});function Rb(e,t,n=0){return y(t.length===2&&t[0].length===2&&t[1].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),tn(e,t,n)}const Rh=v({pad2d_:Rb});function Bb(e,t,n=0){return y(t.length===3&&t[0].length===2&&t[1].length===2&&t[2].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),tn(e,t,n)}const ya=v({pad3d_:Bb});function Pb(e,t,n=0){return y(t.length===4&&t[0].length===2&&t[1].length===2&&t[2].length===2&&t[3].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),tn(e,t,n)}const Bh=v({pad4d_:Pb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Cb(e,t,n){const r=m(e,"x","spaceToBatchND");y(r.rank>=1+t.length,()=>`input rank ${r.rank} should be > than [blockShape] ${t.length}`),y(n.length===t.length,()=>`paddings.shape[0] ${n.length} must be equal to [blockShape] ${t.length}`),y(r.shape.reduce((o,i,u)=>u>0&&u<=t.length?o&&(i+n[u-1][0]+n[u-1][1])%t[u-1]===0:o,!0),()=>`input spatial dimensions ${r.shape.slice(1)} with paddings ${n.toString()} must be divisible by blockShapes ${t.toString()}`);const s={x:r},a={blockShape:t,paddings:n};return S.runKernel(Nl,s,a)}const ba=v({spaceToBatchND_:Cb});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Lb(e,t,n,r,s,a,o){s==null&&(s=[1,1]),a==null&&(a=1),r===0&&(r="valid");const i=m(e,"x","maxPool");let u=i,l=!1;i.rank===3&&(l=!0,u=O(i,[1,i.shape[0],i.shape[1],i.shape[2]])),y(ne(a,s),()=>`Error in pool: Either strides or dilations must be 1. Got strides ${a} and dilations '${s}'`);const h=xc(u.shape,t,a,s,r),c=[h.dilationHeight,h.dilationWidth];let p;r==="same"?p=Mb([h.filterHeight,h.filterWidth],c):p=[[0,0],[0,0]];const d=c[0]===1&&c[1]===1,[g,N]=zb([h.inHeight,h.inWidth],c,p),w=d?r:"valid",T=d?u:ba(u,c,g),$=(n==="avg"?()=>Js(T,t,a,w,o):()=>da(T,t,a,w,o))(),E=d?$:Qs($,c,N);return l?O(E,[E.shape[1],E.shape[2],E.shape[3]]):E}function zb(e,t,n){const r=n.map(h=>h[0]),s=n.map(h=>h[1]),a=e.concat(r,s),o=t.map((h,c)=>(h-a[c]%h)%h),i=s.map((h,c)=>h+o[c]),u=t.map((h,c)=>[r[c],i[c]]),l=t.map((h,c)=>[0,o[c]]);return[u,l]}function Mb(e,t){const r=e.map((o,i)=>o+(o-1)*(t[i]-1)).map(o=>o-1),s=r.map(o=>Math.floor(o/2)),a=r.map((o,i)=>o-s[i]);return r.map((o,i)=>[s[i],a[i]])}const Ph=v({pool_:Lb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vb(e,t){const n=m(e,"x","prelu"),r=m(t,"alpha","prelu"),s={x:n,alpha:r};return S.runKernel(Hu,s)}const wa=v({prelu_:Vb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Wb(e,t=null,n=!1){let r=m(e,"x","prod");r.dtype==="bool"&&(r=j(r,"int32"));const s={x:r},a={axis:t,keepDims:n};return S.runKernel(ju,s,a)}const Ch=v({prod_:Wb});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ub(e,t,n,r){const s=e.map((h,c)=>m(h,`tensors${c}`,"raggedGather","int32")),a=m(t,"paramsDenseValues","raggedGather"),o=m(n,"indices","raggedGather","int32"),i={paramsNestedSplits:s,paramsDenseValues:a,indices:o},u={outputRaggedRank:r},l=S.runKernel(Gu,i,u);return{outputNestedSplits:l.slice(0,l.length-1),outputDenseValues:l[l.length-1]}}const Lh=v({raggedGather_:Ub});/**
 * @license
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qb(e,t,n){const r=m(e,"starts","raggedRange"),s=m(t,"limits","raggedRange",r.dtype),a=m(n,"deltas","raggedRange",r.dtype),o={starts:r,limits:s,deltas:a},i=S.runKernel(Ku,o);return{rtNestedSplits:i[0],rtDenseValues:i[1]}}const zh=v({raggedRange_:qb});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hb(e,t,n,r,s){const a=m(e,"shape","raggedTensorToTensor","int32"),o=m(t,"values","raggedTensorToTensor"),i=m(n,"defaultValue","raggedTensorToTensor",o.dtype),u=r.map((c,p)=>m(c,`tensors${p}`,"raggedTensorToTensor","int32")),l={shape:a,values:o,defaultValue:i,rowPartitionTensors:u},h={rowPartitionTypes:s};return S.runKernel(Xu,l,h)}const Mh=v({raggedTensorToTensor_:Hb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function jb(e,t,n){$t(e);const r=K(e);let s=null;if(n==null||n==="float32")s=new Float32Array(r);else if(n==="int32")s=new Int32Array(r);else if(n==="bool")s=new Uint8Array(r);else throw new Error(`Unknown data type ${n}`);for(let a=0;a<r;a++)s[a]=t();return S.makeTensor(s,e,n)}const Vh=v({rand_:jb});var er={exports:{}},Gb=er.exports,no;function Kb(){return no||(no=1,function(e){(function(t,n,r){function s(u){var l=this,h=i();l.next=function(){var c=2091639*l.s0+l.c*23283064365386963e-26;return l.s0=l.s1,l.s1=l.s2,l.s2=c-(l.c=c|0)},l.c=1,l.s0=h(" "),l.s1=h(" "),l.s2=h(" "),l.s0-=h(u),l.s0<0&&(l.s0+=1),l.s1-=h(u),l.s1<0&&(l.s1+=1),l.s2-=h(u),l.s2<0&&(l.s2+=1),h=null}function a(u,l){return l.c=u.c,l.s0=u.s0,l.s1=u.s1,l.s2=u.s2,l}function o(u,l){var h=new s(u),c=l&&l.state,p=h.next;return p.int32=function(){return h.next()*4294967296|0},p.double=function(){return p()+(p()*2097152|0)*11102230246251565e-32},p.quick=p,c&&(typeof c=="object"&&a(c,h),p.state=function(){return a(h,{})}),p}function i(){var u=4022871197,l=function(h){h=String(h);for(var c=0;c<h.length;c++){u+=h.charCodeAt(c);var p=.02519603282416938*u;u=p>>>0,p-=u,p*=u,u=p>>>0,p-=u,u+=p*4294967296}return(u>>>0)*23283064365386963e-26};return l}n&&n.exports?n.exports=o:this.alea=o})(Gb,e)}(er)),er.exports}var nr={exports:{}},Xb=nr.exports,ro;function Yb(){return ro||(ro=1,function(e){(function(t,n,r){function s(i){var u=this,l="";u.x=0,u.y=0,u.z=0,u.w=0,u.next=function(){var c=u.x^u.x<<11;return u.x=u.y,u.y=u.z,u.z=u.w,u.w^=u.w>>>19^c^c>>>8},i===(i|0)?u.x=i:l+=i;for(var h=0;h<l.length+64;h++)u.x^=l.charCodeAt(h)|0,u.next()}function a(i,u){return u.x=i.x,u.y=i.y,u.z=i.z,u.w=i.w,u}function o(i,u){var l=new s(i),h=u&&u.state,c=function(){return(l.next()>>>0)/4294967296};return c.double=function(){do var p=l.next()>>>11,d=(l.next()>>>0)/4294967296,g=(p+d)/(1<<21);while(g===0);return g},c.int32=l.next,c.quick=c,h&&(typeof h=="object"&&a(h,l),c.state=function(){return a(l,{})}),c}n&&n.exports?n.exports=o:this.xor128=o})(Xb,e)}(nr)),nr.exports}var rr={exports:{}},Zb=rr.exports,so;function Jb(){return so||(so=1,function(e){(function(t,n,r){function s(i){var u=this,l="";u.next=function(){var c=u.x^u.x>>>2;return u.x=u.y,u.y=u.z,u.z=u.w,u.w=u.v,(u.d=u.d+362437|0)+(u.v=u.v^u.v<<4^(c^c<<1))|0},u.x=0,u.y=0,u.z=0,u.w=0,u.v=0,i===(i|0)?u.x=i:l+=i;for(var h=0;h<l.length+64;h++)u.x^=l.charCodeAt(h)|0,h==l.length&&(u.d=u.x<<10^u.x>>>4),u.next()}function a(i,u){return u.x=i.x,u.y=i.y,u.z=i.z,u.w=i.w,u.v=i.v,u.d=i.d,u}function o(i,u){var l=new s(i),h=u&&u.state,c=function(){return(l.next()>>>0)/4294967296};return c.double=function(){do var p=l.next()>>>11,d=(l.next()>>>0)/4294967296,g=(p+d)/(1<<21);while(g===0);return g},c.int32=l.next,c.quick=c,h&&(typeof h=="object"&&a(h,l),c.state=function(){return a(l,{})}),c}n&&n.exports?n.exports=o:this.xorwow=o})(Zb,e)}(rr)),rr.exports}var sr={exports:{}},Qb=sr.exports,ao;function t0(){return ao||(ao=1,function(e){(function(t,n,r){function s(i){var u=this;u.next=function(){var h=u.x,c=u.i,p,d;return p=h[c],p^=p>>>7,d=p^p<<24,p=h[c+1&7],d^=p^p>>>10,p=h[c+3&7],d^=p^p>>>3,p=h[c+4&7],d^=p^p<<7,p=h[c+7&7],p=p^p<<13,d^=p^p<<9,h[c]=d,u.i=c+1&7,d};function l(h,c){var p,d=[];if(c===(c|0))d[0]=c;else for(c=""+c,p=0;p<c.length;++p)d[p&7]=d[p&7]<<15^c.charCodeAt(p)+d[p+1&7]<<13;for(;d.length<8;)d.push(0);for(p=0;p<8&&d[p]===0;++p);for(p==8?d[7]=-1:d[p],h.x=d,h.i=0,p=256;p>0;--p)h.next()}l(u,i)}function a(i,u){return u.x=i.x.slice(),u.i=i.i,u}function o(i,u){i==null&&(i=+new Date);var l=new s(i),h=u&&u.state,c=function(){return(l.next()>>>0)/4294967296};return c.double=function(){do var p=l.next()>>>11,d=(l.next()>>>0)/4294967296,g=(p+d)/(1<<21);while(g===0);return g},c.int32=l.next,c.quick=c,h&&(h.x&&a(h,l),c.state=function(){return a(l,{})}),c}n&&n.exports?n.exports=o:this.xorshift7=o})(Qb,e)}(sr)),sr.exports}var ar={exports:{}},e0=ar.exports,oo;function n0(){return oo||(oo=1,function(e){(function(t,n,r){function s(i){var u=this;u.next=function(){var h=u.w,c=u.X,p=u.i,d,g;return u.w=h=h+1640531527|0,g=c[p+34&127],d=c[p=p+1&127],g^=g<<13,d^=d<<17,g^=g>>>15,d^=d>>>12,g=c[p]=g^d,u.i=p,g+(h^h>>>16)|0};function l(h,c){var p,d,g,N,w,T=[],x=128;for(c===(c|0)?(d=c,c=null):(c=c+"\0",d=0,x=Math.max(x,c.length)),g=0,N=-32;N<x;++N)c&&(d^=c.charCodeAt((N+32)%c.length)),N===0&&(w=d),d^=d<<10,d^=d>>>15,d^=d<<4,d^=d>>>13,N>=0&&(w=w+1640531527|0,p=T[N&127]^=d+w,g=p==0?g+1:0);for(g>=128&&(T[(c&&c.length||0)&127]=-1),g=127,N=4*128;N>0;--N)d=T[g+34&127],p=T[g=g+1&127],d^=d<<13,p^=p<<17,d^=d>>>15,p^=p>>>12,T[g]=d^p;h.w=w,h.X=T,h.i=g}l(u,i)}function a(i,u){return u.i=i.i,u.w=i.w,u.X=i.X.slice(),u}function o(i,u){i==null&&(i=+new Date);var l=new s(i),h=u&&u.state,c=function(){return(l.next()>>>0)/4294967296};return c.double=function(){do var p=l.next()>>>11,d=(l.next()>>>0)/4294967296,g=(p+d)/(1<<21);while(g===0);return g},c.int32=l.next,c.quick=c,h&&(h.X&&a(h,l),c.state=function(){return a(l,{})}),c}n&&n.exports?n.exports=o:this.xor4096=o})(e0,e)}(ar)),ar.exports}var or={exports:{}},r0=or.exports,io;function s0(){return io||(io=1,function(e){(function(t,n,r){function s(i){var u=this,l="";u.next=function(){var c=u.b,p=u.c,d=u.d,g=u.a;return c=c<<25^c>>>7^p,p=p-d|0,d=d<<24^d>>>8^g,g=g-c|0,u.b=c=c<<20^c>>>12^p,u.c=p=p-d|0,u.d=d<<16^p>>>16^g,u.a=g-c|0},u.a=0,u.b=0,u.c=-1640531527,u.d=1367130551,i===Math.floor(i)?(u.a=i/4294967296|0,u.b=i|0):l+=i;for(var h=0;h<l.length+20;h++)u.b^=l.charCodeAt(h)|0,u.next()}function a(i,u){return u.a=i.a,u.b=i.b,u.c=i.c,u.d=i.d,u}function o(i,u){var l=new s(i),h=u&&u.state,c=function(){return(l.next()>>>0)/4294967296};return c.double=function(){do var p=l.next()>>>11,d=(l.next()>>>0)/4294967296,g=(p+d)/(1<<21);while(g===0);return g},c.int32=l.next,c.quick=c,h&&(typeof h=="object"&&a(h,l),c.state=function(){return a(l,{})}),c}n&&n.exports?n.exports=o:this.tychei=o})(r0,e)}(or)),or.exports}var ir={exports:{}};const a0={},o0=Object.freeze(Object.defineProperty({__proto__:null,default:a0},Symbol.toStringTag,{value:"Module"})),i0=_d(o0);var u0=ir.exports,uo;function l0(){return uo||(uo=1,function(e){(function(t,n,r){var s=256,a=6,o=52,i="random",u=r.pow(s,a),l=r.pow(2,o),h=l*2,c=s-1,p;function d(E,I,A){var F=[];I=I==!0?{entropy:!0}:I||{};var R=T(w(I.entropy?[E,$(n)]:E??x(),3),F),k=new g(F),_=function(){for(var b=k.g(a),D=u,P=0;b<l;)b=(b+P)*s,D*=s,P=k.g(1);for(;b>=h;)b/=2,D/=2,P>>>=1;return(b+P)/D};return _.int32=function(){return k.g(4)|0},_.quick=function(){return k.g(4)/4294967296},_.double=_,T($(k.S),n),(I.pass||A||function(b,D,P,C){return C&&(C.S&&N(C,k),b.state=function(){return N(k,{})}),P?(r[i]=b,D):b})(_,R,"global"in I?I.global:this==r,I.state)}function g(E){var I,A=E.length,F=this,R=0,k=F.i=F.j=0,_=F.S=[];for(A||(E=[A++]);R<s;)_[R]=R++;for(R=0;R<s;R++)_[R]=_[k=c&k+E[R%A]+(I=_[R])],_[k]=I;(F.g=function(b){for(var D,P=0,C=F.i,L=F.j,q=F.S;b--;)D=q[C=c&C+1],P=P*s+q[c&(q[C]=q[L=c&L+D])+(q[L]=D)];return F.i=C,F.j=L,P})(s)}function N(E,I){return I.i=E.i,I.j=E.j,I.S=E.S.slice(),I}function w(E,I){var A=[],F=typeof E,R;if(I&&F=="object")for(R in E)try{A.push(w(E[R],I-1))}catch{}return A.length?A:F=="string"?E:E+"\0"}function T(E,I){for(var A=E+"",F,R=0;R<A.length;)I[c&R]=c&(F^=I[c&R]*19)+A.charCodeAt(R++);return $(I)}function x(){try{var E;return p&&(E=p.randomBytes)?E=E(s):(E=new Uint8Array(s),(t.crypto||t.msCrypto).getRandomValues(E)),$(E)}catch{var I=t.navigator,A=I&&I.plugins;return[+new Date,t,A,t.screen,$(n)]}}function $(E){return String.fromCharCode.apply(0,E)}if(T(r.random(),n),e.exports){e.exports=d;try{p=i0}catch{}}else r["seed"+i]=d})(typeof self<"u"?self:u0,[],Math)}(ir)),ir.exports}var qr,lo;function c0(){if(lo)return qr;lo=1;var e=Kb(),t=Yb(),n=Jb(),r=t0(),s=n0(),a=s0(),o=l0();return o.alea=e,o.xor128=t,o.xorwow=n,o.xorshift7=r,o.xor4096=s,o.tychei=a,qr=o,qr}var Na=c0();/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const h0=.001,Wh=.1;function p0(e,t,n){return n==null&&(n=va()),ms(e,t,(r,s)=>Sa(r,s,n))}function va(){return S.backend.floatPrecision()===32?h0:Wh}function ms(e,t,n){let r=!0;if((ut(e)||ut(t))&&(r=!1),ut(e)&&ut(t)&&(r=!0),r){const o=e.constructor.name,i=t.constructor.name;if(o!==i)throw new Error(`Arrays are of different type. Actual: ${o}. Expected: ${i}`)}if(Array.isArray(e)&&Array.isArray(t)){const o=Ut(e),i=Ut(t);if(!Pt(o,i))throw new Error(`Arrays have different shapes. Actual: [${o}]. Expected: [${i}]`)}const s=ut(e)?e:pe(e),a=ut(t)?t:pe(t);if(s.length!==a.length)throw new Error(`Arrays have different lengths actual: ${s.length} vs expected: ${a.length}.
Actual:   ${s}.
Expected: ${a}.`);for(let o=0;o<a.length;++o){const i=s[o],u=a[o];if(!n(i,u))throw new Error(`Arrays differ: actual[${o}] = ${i}, expected[${o}] = ${u}.
Actual:   ${s}.
Expected: ${a}.`)}typeof expect<"u"&&expect().nothing()}function f0(e,t){e().then(()=>t.fail(),()=>t()),typeof expect<"u"&&expect().nothing()}function d0(e,t){const n=typeof t=="string"||typeof t=="number"||typeof t=="boolean"?[t]:t;return oe(e)||oe(e[0])||oe(t)||oe(t[0])?ms(e,n,(r,s)=>r==s):ms(e,t,(r,s)=>Sa(r,s,0))}function m0(e,t,n){if(n==null&&(n=va()),!Sa(e,t,n))throw new Error(`Numbers differ: actual === ${e}, expected === ${t}`);typeof expect<"u"&&expect().nothing()}function Sa(e,t,n){return!isFinite(e)&&!isFinite(t)?!0:!(isNaN(e)||isNaN(t)||Math.abs(e-t)>n)}function g0(e,t,n){for(let r=0;r<e.length;r++)if(e[r]<t||e[r]>n)throw new Error(`Value out of range:${e[r]} low: ${t}, high: ${n}`)}function y0(e,t){const n=new Float32Array(e),r=new Float32Array(t);if(n.length!==r.length)throw new Error(`Expected ArrayBuffer to be of length ${r.length}, but it was ${n.length}`);for(let s=0;s<r.length;s++)if(n[s]!==r[s])throw new Error(`Expected ArrayBuffer value at ${s} to be ${r[s]} but got ${n[s]} instead`)}function Uh(e){for(let t=0;t<e.length;t++){const n=e[t];Array.isArray(n)?Uh(n):e[t]=xn(n)}return e}function b0(e){const t=document.createElement("video");return"playsInline"in t&&(t.playsInline=!0),t.muted=!0,t.loop=!0,t.style.position="fixed",t.style.left="0px",t.style.top="0px",t.preload="auto",t.appendChild(e),new Promise(n=>{t.addEventListener("loadeddata",r=>n(t)),t.load()})}async function w0(e){await e.play(),"requestVideoFrameCallback"in e&&await new Promise(t=>{e.requestVideoFrameCallback(t)})}const N0=Object.freeze(Object.defineProperty({__proto__:null,TEST_EPSILON_FLOAT16:Wh,createVideoElement:b0,encodeStrings:Uh,expectArrayBuffersEqual:y0,expectArraysClose:p0,expectArraysEqual:d0,expectNumbersClose:m0,expectPromiseToFail:f0,expectValuesInRange:g0,play:w0,testEpsilon:va},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Ta{constructor(t,n,r,s,a){this.mean=t,this.stdDev=n,this.dtype=r,this.nextVal=NaN,this.truncated=s,this.truncated&&(this.upper=this.mean+this.stdDev*2,this.lower=this.mean-this.stdDev*2);const o=a||Math.random();this.random=Na.alea(o.toString())}nextValue(){if(!isNaN(this.nextVal)){const s=this.nextVal;return this.nextVal=NaN,s}let t,n,r=!1;for(;!r;){let s,a,o;do s=2*this.random()-1,a=2*this.random()-1,o=s*s+a*a;while(o>=1||o===0);const i=Math.sqrt(-2*Math.log(o)/o);t=this.mean+this.stdDev*s*i,n=this.mean+this.stdDev*a*i,(!this.truncated||this.isValidTruncated(t))&&(r=!0)}return(!this.truncated||this.isValidTruncated(n))&&(this.nextVal=this.convertValue(n)),this.convertValue(t)}convertValue(t){return this.dtype==null||this.dtype==="float32"?t:Math.round(t)}isValidTruncated(t){return t<=this.upper&&t>=this.lower}}class v0{constructor(t,n,r,s){this.alpha=t,this.beta=1/n,this.dtype=r;const a=s||Math.random();this.randu=Na.alea(a.toString()),this.randn=new Ta(0,1,r,!1,this.randu()),t<1?this.d=t+2/3:this.d=t-1/3,this.c=1/Math.sqrt(9*this.d)}nextValue(){let t,n,r,s,a,o;for(;;){do s=this.randn.nextValue(),o=1+this.c*s;while(o<=0);if(o*=o*o,t=s*s,n=1-.331*t*t,r=.5*t+this.d*(1-o+Math.log(o)),a=this.randu(),a<n||Math.log(a)<r)break}return o=1/this.beta*this.d*o,this.alpha<1&&(o*=Math.pow(this.randu(),1/this.alpha)),this.convertValue(o)}convertValue(t){return this.dtype==="float32"?t:Math.round(t)}}class S0{constructor(t=0,n=1,r,s){if(this.canReturnFloat=()=>this.dtype==null||this.dtype==="float32",this.min=t,this.range=n-t,this.dtype=r,s==null&&(s=Math.random()),typeof s=="number"&&(s=s.toString()),!this.canReturnFloat()&&this.range<=1)throw new Error(`The difference between ${t} - ${n} <= 1 and dtype is not float`);this.random=Na.alea(s)}convertValue(t){return this.canReturnFloat()?t:Math.round(t)}nextValue(){return this.convertValue(this.min+this.range*this.random())}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function T0(e,t,n=1,r="float32",s){if($t(e),n==null&&(n=1),r==null&&(r="float32"),r!=="float32"&&r!=="int32")throw new Error(`Unsupported data type ${r}`);const a=new v0(t,n,r,s),o=qt(e,r);for(let i=0;i<o.values.length;i++)o.values[i]=a.nextValue();return o.toTensor()}const qh=v({randomGamma_:T0});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function E0(e,t=0,n=1,r,s){if($t(e),r!=null&&r==="bool")throw new Error(`Unsupported data type ${r}`);const a=new Ta(t,n,r,!1,s),o=qt(e,r);for(let i=0;i<o.values.length;i++)o.values[i]=a.nextValue();return o.toTensor()}const Ea=v({randomNormal_:E0});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $0(e,t,n){if(t!=null&&t==="bool")throw new Error(`Unsupported data type ${t}`);return Ea(e,0,1,t,n)}const Hh=v({randomStandardNormal_:$0});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _0(e,t=0,n=1,r="float32",s){$t(e);const a=qt(e,r),o=new S0(t,n,null,s);for(let i=0;i<a.values.length;i++)a.values[i]=o.nextValue();return a.toTensor()}const Ir=v({randomUniform_:_0});/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function k0(e,t,n,r){return Ir(e,t,n,"int32",r)}const jh=v({randomUniformInt_:k0});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function me(e,t,n=1,r="float32"){if(n===0)throw new Error("Cannot have a step of zero");const s={start:e,stop:t,step:n,dtype:r};return S.runKernel(Yu,{},s)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function I0(e){const n={input:m(e,"input","real")};return S.runKernel(Zu,n)}const Xe=v({real_:I0});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function x0(e){const n={x:m(e,"x","reciprocal")};return S.runKernel(Ju,n)}const Gh=v({reciprocal_:x0});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function A0(e){const n={x:m(e,"x","relu")};return S.runKernel(Qu,n)}const Ln=v({relu_:A0});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function O0(e){const n={x:m(e,"x","relu6")};return S.runKernel(rl,n)}const $a=v({relu6_:O0});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function D0(e,t){const r={x:m(e,"x","reverse")},s={dims:t};return S.runKernel(sl,r,s)}const ge=v({reverse_:D0});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function F0(e){const t=m(e,"x","reverse");return y(t.rank===1,()=>`Error in reverse1D: x must be rank 1 but got rank ${t.rank}.`),ge(t,0)}const Kh=v({reverse1d_:F0});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function R0(e,t){const n=m(e,"x","reverse");return y(n.rank===2,()=>`Error in reverse2D: x must be rank 2 but got rank ${n.rank}.`),ge(n,t)}const Xh=v({reverse2d_:R0});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function B0(e,t){const n=m(e,"x","reverse");return y(n.rank===3,()=>`Error in reverse3D: x must be rank 3 but got rank ${n.rank}.`),ge(n,t)}const Yh=v({reverse3d_:B0});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function P0(e,t){const n=m(e,"x","reverse");return y(n.rank===4,()=>`Error in reverse4D: x must be rank 4 but got rank ${n.rank}.`),ge(n,t)}const Zh=v({reverse4d_:P0});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function C0(e){const n={x:m(e,"x","round")};return S.runKernel(al,n)}const _a=v({round_:C0});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function L0(e){const n={x:m(e,"x","rsqrt","float32")};return S.runKernel(ol,n)}const Jh=v({rsqrt_:L0});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function z0(e){const n={x:m(e,"x","selu")};return S.runKernel(hl,n)}const Qh=v({selu_:z0});function M0(e,t,n,r,s,a=[1,1],o="NHWC"){const i=m(e,"x","separableConv2d"),u=m(t,"depthwiseFilter","separableConv2d"),l=m(n,"pointwiseFilter","separableConv2d");let h=i,c=!1;if(i.rank===3&&(c=!0,h=O(i,[1,i.shape[0],i.shape[1],i.shape[2]])),o==="NCHW")throw new Error("separableConv2d currently does not support dataFormat NCHW; only NHWC is supported");y(h.rank===4,()=>`Error in separableConv2d: input must be rank 4, but got rank ${h.rank}.`),y(u.rank===4,()=>`Error in separableConv2d: depthwise filter must be rank 4, but got rank ${u.rank}.`),y(l.rank===4,()=>`Error in separableConv2d: pointwise filter must be rank 4, but got rank ${u.rank}.`),y(l.shape[0]===1,()=>`Error in separableConv2d: the first dimension of pointwise filter  must be 1, but got ${l.shape[0]}.`),y(l.shape[1]===1,()=>`Error in separableConv2d: the second dimension of pointwise filter must be 1, but got ${l.shape[1]}.`);const p=u.shape[2],d=u.shape[3];y(l.shape[2]===p*d,()=>`Error in separableConv2d: the third dimension of pointwise filter must be ${p*d}, but got ${l.shape[2]}.`);const g=Er(h,u,r,s,o,a),w=Fn(g,l,1,"valid",o);return c?O(w,[w.shape[1],w.shape[2],w.shape[3]]):w}const tp=v({separableConv2d_:M0});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function V0(e,t){const n=m(e,"x","setdiff1d"),r=m(t,"y","setdiff1d");y(n.dtype===r.dtype,()=>`x and y should have the same dtype, but got x (${n.dtype}) and y (${r.dtype}).`),y(n.rank===1,()=>`x should be 1D tensor, but got x (${n.shape}).`),y(r.rank===1,()=>`y should be 1D tensor, but got y (${r.shape}).`);const s=await n.data(),a=await r.data(),o=new Set(a);let i=0;for(let h=0;h<s.length;h++)o.has(s[h])||i++;const u=new dr([i],n.dtype),l=new dr([i],"int32");for(let h=0,c=0;h<s.length;h++)o.has(s[h])||(u.values[c]=s[h],l.values[c]=h,c++);return[u.toTensor(),l.toTensor()]}const ep=V0;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function W0(e){const n={x:m(e,"x","sign")};return S.runKernel(ml,n)}const np=v({sign_:W0});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function U0(e){const n={x:m(e,"x","sin","float32")};return S.runKernel(fl,n)}const rp=v({sin_:U0});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function q0(e){const n={x:m(e,"x","sinh")};return S.runKernel(dl,n)}const sp=v({sinh_:q0});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function H0(e,t,n){const r=m(e,"x","slice1d");return y(r.rank===1,()=>`slice1d expects a rank-1 tensor, but got a rank-${r.rank} tensor`),X(r,[t],[n])}const ap=v({slice1d_:H0});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function j0(e,t,n){const r=m(e,"x","slice2d");return y(r.rank===2,()=>`slice2d expects a rank-2 tensor, but got a rank-${r.rank} tensor`),X(r,t,n)}const op=v({slice2d_:j0});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function G0(e,t,n){const r=m(e,"x","slice3d");return y(r.rank===3,()=>`slice3d expects a rank-3 tensor, but got a rank-${r.rank} tensor`),X(r,t,n)}const ip=v({slice3d_:G0});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function K0(e,t,n){const r=m(e,"x","slice4d");return y(r.rank===4,()=>`slice4d expects a rank-4 tensor, but got a rank-${r.rank} tensor`),X(r,t,n)}const up=v({slice4d_:K0});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function X0(e,t=-1){const n=m(e,"logits","softmax","float32");if(t===-1&&(t=n.rank-1),t!==n.rank-1)throw Error(`Softmax along a non-last dimension is not yet supported. Logits was rank ${n.rank} and dim was ${t}`);const r={logits:n},s={dim:t};return S.runKernel(Sl,r,s)}const lp=v({softmax_:X0});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Y0(e){y(e.dtype==="complex64",()=>`The dtype for tf.spectral.fft() must be complex64 but got ${e.dtype}.`);const t={input:e};return S.runKernel(Ji,t)}const xr=v({fft_:Y0});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Z0(e){y(e.dtype==="complex64",()=>`The dtype for tf.spectral.ifft() must be complex64 but got ${e.dtype}.`);const t={input:e};return S.runKernel(uu,t)}const En=v({ifft_:Z0});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function J0(e){const t=e.shape[e.shape.length-1],n=e.size/t;let r;if(t<=2){const s=O(e,[n,t]);r=En(s)}else{const s=[n,2*(t-1)],a=O(Xe(e),[n,t]),o=O(Cn(e),[n,t]),i=ge(X(a,[0,1],[n,t-2]),1),u=B(ge(X(o,[0,1],[n,t-2]),1),U(-1)),l=ht([a,i],1),h=ht([o,u],1),c=O(ee(l,h),[s[0],s[1]]);r=En(c)}if(r=Xe(r),e.rank===3&&e.shape[0]!==0){const s=r,a=e.shape[0];r=O(r,[a,r.shape[0]/a,r.shape[1]]),s.dispose()}return r}const ka=v({irfft_:J0});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Q0(e,t,n=0){const s={x:m(e,"x","split")},a={numOrSizeSplits:t,axis:n};return S.runKernel(vl,s,a)}const Ye=v({split_:Q0});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function tw(e,t){y(e.dtype==="float32",()=>`The dtype for rfft() must be real value but got ${e.dtype}`);let n=e.shape[e.shape.length-1];const r=e.size/n;let s;if(t!=null&&t<n){const g=e.shape.map(w=>0),N=e.shape.map(w=>w);N[e.shape.length-1]=t,s=X(e,g,N),n=t}else if(t!=null&&t>n){const g=e.shape.map(N=>N);g[e.shape.length-1]=t-n,s=ht([e,De(g)],e.shape.length-1),n=t}else s=e;const a=Et(s),o=O(ee(s,a),[r,n]),i=xr(o),u=Math.floor(n/2)+1,l=Xe(i),h=Cn(i),c=Ye(l,[u,n-u],l.shape.length-1),p=Ye(h,[u,n-u],h.shape.length-1),d=s.shape.slice();return d[s.shape.length-1]=u,O(ee(c[0],p[0]),d)}const Ar=v({rfft_:tw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ew(e,t){let n=m(e,"a","squaredDifference"),r=m(t,"b","squaredDifference");[n,r]=st(n,r),at(n.shape,r.shape);const s={a:n,b:r},a={};return S.runKernel(Il,s,a)}const Ia=v({squaredDifference_:ew});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function nw(e,t){const n=m(e,"x","squeeze","string_or_numeric");return O(n,jo(n.shape,t).newShape)}const Mt=v({squeeze_:nw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function rw(e,t=0){const n=gn(e,"tensors","stack","string_or_numeric");y(n.length>=1,()=>"Pass at least one tensor to tf.stack"),n.length>0&&y(t<=n[0].rank,()=>"Axis must be <= rank of the tensor");const r=n,s={axis:t};return S.runKernel(Wu,r,s)}const Gt=v({stack_:rw});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function sw(e,t=0){const r={x:m(e,"x","step")},s={alpha:t};return S.runKernel(Ul,r,s)}const xa=v({step_:sw});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function aw(e,t,n,r,s=0,a=0,o=0,i=0,u=0){const h={x:m(e,"x","stridedSlice","string_or_numeric")},c={begin:t,end:n,strides:r,beginMask:s,endMask:a,ellipsisMask:o,newAxisMask:i,shrinkAxisMask:u};return S.runKernel(Al,h,c)}const cp=v({stridedSlice_:aw});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ow(e){const n={x:m(e,"x","tan","float32")};return S.runKernel(Bl,n)}const hp=v({tan_:ow});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function kt(e,t){Be(e);const n=Ut(e,t);if(n.length!==1)throw new Error("tensor1d() requires values to be a flat/TypedArray");return ye(e,null,n,t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ue(e,t,n){if(Be(e),t!=null&&t.length!==2)throw new Error("tensor2d() requires shape to have two numbers");const r=Ut(e,n);if(r.length!==2&&r.length!==1)throw new Error("tensor2d() requires values to be number[][] or flat/TypedArray");if(r.length===1&&t==null)throw new Error("tensor2d() requires shape to be provided when `values` are a flat/TypedArray");return ye(e,t,r,n)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Aa(e,t,n){if(Be(e),t!=null&&t.length!==3)throw new Error("tensor3d() requires shape to have three numbers");const r=Ut(e,n);if(r.length!==3&&r.length!==1)throw new Error("tensor3d() requires values to be number[][][] or flat/TypedArray");if(r.length===1&&t==null)throw new Error("tensor3d() requires shape to be provided when `values` are a flat array");return ye(e,t,r,n)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function pp(e,t,n){if(Be(e),t!=null&&t.length!==4)throw new Error("tensor4d() requires shape to have four numbers");const r=Ut(e,n);if(r.length!==4&&r.length!==1)throw new Error("tensor4d() requires values to be number[][][][] or flat/TypedArray");if(r.length===1&&t==null)throw new Error("tensor4d() requires shape to be provided when `values` are a flat array");return ye(e,t,r,n)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fp(e,t,n){if(Be(e),t!=null&&t.length!==5)throw new Error("tensor5d() requires shape to have five numbers");const r=Ut(e,n);if(r.length!==5&&r.length!==1)throw new Error("tensor5d() requires values to be number[][][][][] or flat/TypedArray");if(r.length===1&&t==null)throw new Error("tensor5d() requires shape to be provided when `values` are a flat array");return ye(e,t,r,n)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function dp(e,t,n){if(Be(e),t!=null&&t.length!==6)throw new Error("tensor6d() requires shape to have six numbers");const r=Ut(e,n);if(r.length!==6&&r.length!==1)throw new Error("tensor6d() requires values to be number[][][][][][] or flat/TypedArray");if(r.length===1&&t==null)throw new Error("tensor6d() requires shape to be provided when `values` are a flat array");return t=t||r,ye(e,t,r,n)}function Oa(e,t,n){const r=t.rank>1?t.shape[t.rank-1]:1,s=t.rank>1?t.rank-1:1,a=`Must have updates.shape = indices.shape[:batchDim] + shape[sliceDim:], got updates.shape: ${n.shape}, indices.shape: ${t.shape}, shape: ${e}, sliceDim: ${r}, and batchDim: ${s}.`;if(n.rank<s)throw new Error(a+` update.rank < ${s}. `);if(e.length<r+(n.rank-s))throw new Error(a+` Output shape length < ${r+(n.rank-s)}`);if(n.rank!==s+e.length-r)throw new Error(a+` update.rank != ${s+e.length-r}`);for(let o=0;o<s;++o)if(n.shape[o]!==t.shape[o])throw new Error(a+` updates.shape[${o}] (${n.shape[o]}) != indices.shape[${o}] (${t.shape[o]}).`);for(let o=0;o<n.rank-s;++o)if(n.shape[o+s]!==e[o+r])throw new Error(a+` updates.shape[${o+s}] (${n.shape[o+s]}) != shape[${o+s}] (${e[o+s]})`)}function Or(e,t,n){if(t.rank<1)throw new Error(`tf.scatterND() expects the indices to be rank 1 or higher, but the rank was ${t.rank}.`);if(e.rank<1)throw new Error(`tf.scatterND() expects the updates to be rank 1 or higher, but the rank was ${e.rank}.`);if(t.dtype!=="int32")throw new Error(`The dtype of 'indices' should be int32, but got dtype: ${t.dtype}`);if(n.length<1)throw new Error(`Output rank must be greater or equal to 1, but got shape: ${n}`);if(n.length===0){if(t.size===0)throw new Error(`Indices specified for empty output. indices shape: ${t.shape}`);if(e.size===0)throw new Error(`Updates specified for empty output. updates shape: ${e.shape}`)}Oa(n,t,e)}function mp(e,t,n){const r=t.shape.length,s=r>1?t.shape[r-1]:1,a=n.length;let o=1;for(let c=s;c<a;++c)o*=n[c];const i=s<1?1:s,u=K(t.shape)/i,l=[...Ze(n.slice(0,s)),1],h=K(n);return{sliceRank:s,numUpdates:u,sliceSize:o,strides:l,outputSize:h}}const iw=Object.freeze(Object.defineProperty({__proto__:null,calculateShapes:mp,validateInput:Or,validateUpdateShape:Oa},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function uw(e,t,n){const r=m(e,"tensor","tensorScatterupdate"),s=m(t,"indices","tensorScatterupdate","int32"),a=m(n,"updates","tensorScatterupdate");if(Or(a,s,r.shape),r.dtype!==a.dtype)throw new Error(`tensor and updates must have the same dtype, instead they are ${r.dtype} and ${a.dtype}.`);const o={tensor:r,indices:s,updates:a},i={};return S.runKernel(ul,o,i)}const gp=v({tensorScatterUpdate_:uw});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function lw(e,t=1,n=!0){const r=m(e,"x","topk");if(r.rank===0)throw new Error("topk() expects the input to be of rank 1 or higher");const s=r.shape[r.shape.length-1];if(t<0)throw new Error(`'k' passed to topk() must be >= 0 but got ${t}`);if(t>s)throw new Error(`'k' passed to topk() must be <= the last dimension (${s}) but got ${t}`);const a={x:r},o={k:t,sorted:n},[i,u]=S.runKernel(Cl,a,o);return{values:i,indices:u}}const yp=v({topk_:lw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function cw(e,t=0,n=1,r,s){if($t(e),r!=null&&r==="bool")throw new Error("Unsupported data type $ { dtype }");const a=new Ta(t,n,r,!0,s),o=qt(e,r);for(let i=0;i<o.values.length;i++)o.values[i]=a.nextValue();return o.toTensor()}const bp=v({truncatedNormal_:cw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hw(e,t=0){const n=m(e,"x","unique","string_or_numeric");y(n.rank>0,()=>"The input tensor must be at least 1D");const r={x:n},s={axis:t},[a,o]=S.runKernel(zl,r,s);return{values:a,indices:o}}const wp=v({unique_:hw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function pw(e,t,n){const r=m(e,"x","unsortedSegmentSum"),s=m(t,"segmentIds","unsortedSegmentSum","int32");y(qe(n),()=>"numSegments must be of dtype int");const a={x:r,segmentIds:s},o={numSegments:n};return S.runKernel(Vl,a,o)}const Np=v({unsortedSegmentSum_:pw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fw(e,t=0){const n=m(e,"x","unstack","string_or_numeric");y(t>=-n.shape.length&&t<n.shape.length,()=>`Axis = ${t} is not in [-${n.shape.length}, ${n.shape.length})`);const r={value:n},s={axis:t};return S.runKernel(Ml,r,s)}const be=v({unstack_:fw});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vp(e,t){return kr(e,t,"right")}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Sp(e,t=!0,n,r){return S.makeVariable(e,t,n,r)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Tp(e,t){const n=[];for(let a=0;a<t.length;a++)t[a]&&n.push(a);const r=qt(e,"int32"),s=qt([n.length,e.length],"int32");for(let a=0;a<n.length;a++){const o=r.indexToLoc(n[a]),i=a*e.length;s.values.set(o,i)}return s.toTensor()}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function dw(e){const t=m(e,"condition","whereAsync","bool"),n=await t.data(),r=Tp(t.shape,n);return e!==t&&t.dispose(),r}const Da=dw;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function mw(e,t,n){const r=m(e,"tensor","boolMask"),s=m(t,"mask","boolMask","bool"),a=n??0,o=s.rank,i=r.shape;y(o>0,()=>"mask cannot be scalar"),gt(i.slice(a,a+o),s.shape,"mask's shape must match the first K dimensions of tensor's shape,");let u=1;for(let N=a;N<a+o;N++)u*=i[N];const l=i.slice(0,a).concat([u],i.slice(a+o)),h=O(r,l),c=O(s,[-1]),p=await Da(c),d=Mt(p,[1]),g=oa(h,d,a);return e!==r&&r.dispose(),t!==s&&s.dispose(),d.dispose(),h.dispose(),c.dispose(),p.dispose(),g}const Ep=mw;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function gw(e,t,n){const r=m(e,"x","transpose");if(t==null&&(t=r.shape.map((o,i)=>i).reverse()),y(r.rank===t.length,()=>`Error in transpose: rank of input ${r.rank} must match length of perm ${t}.`),t.forEach(o=>{y(o>=0&&o<r.rank,()=>`All entries in 'perm' must be between 0 and ${r.rank-1} but got ${t}`)}),r.rank<=1)return r.clone();const s={x:r},a={perm:t};return r.dtype==="complex64"?V(()=>{let o=Xe(r),i=Cn(r);return o=S.runKernel(Jn,{x:o},a),i=S.runKernel(Jn,{x:i},a),n&&(i=Bt(i)),ee(o,i)}):S.runKernel(Jn,s,a)}const $n=v({transpose_:gw});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function yw(e,t,n,r,s=!0){const a=m(e,"v","movingAverage"),o=m(t,"x","movingAverage"),i=m(n,"decay","movingAverage");ec(a,o),y(Pt(a.shape,o.shape),()=>"Shape mismatch in v and x");const u=U(1),l=W(u,i);let h=B(W(o,a),l);if(s){y(r!=null,()=>"When using zeroDebias: true, step is required.");const c=m(r,"step","movingAverage");h=Y(h,W(u,Ge(i,c)))}return z(a,h)}const $p=v({movingAverage_:yw});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function bw(e,t,n){$t(n);const r=m(e,"indices","scatterND","int32"),s=m(t,"updates","scatterND");Or(s,r,n);const a={indices:r,updates:s},o={shape:n};return S.runKernel(il,a,o)}const _p=v({scatterND_:bw});function ww(e,t,n,r){if(e.dtype!=="int32")throw new Error(`tf.sparseToDense() expects the indices to be int32 type, but the dtype was ${e.dtype}.`);if(e.rank>2)throw new Error(`sparseIndices should be a scalar, vector, or matrix, but got shape ${e.shape}.`);const s=e.rank>0?e.shape[0]:1,a=e.rank>1?e.shape[1]:1;if(n.length!==a)throw new Error(`outputShape has incorrect number of elements:, ${n.length}, should be: ${a}.`);const o=t.size;if(!(t.rank===0||t.rank===1&&o===s))throw new Error(`sparseValues has incorrect shape ${t.shape}, should be [] or [${s}]`);if(t.dtype!==r.dtype)throw new Error("sparseValues.dtype must match defaultValues.dtype")}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Nw(e,t,n,r=0){$t(n);const s=m(e,"sparseIndices","sparseToDense","int32"),a=m(t,"sparseValues","sparseToDense","string_or_numeric"),o=m(r,"defaultValue","sparseToDense",a.dtype);ww(s,a,n,o);const i={sparseIndices:s,sparseValues:a,defaultValue:o},u={outputShape:n};return S.runKernel(kl,i,u)}const kp=v({sparseToDense_:Nw});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vw(e,t){const n=m(t,"indices","gatherND","int32"),s={params:m(e,"x","gatherND","string_or_numeric"),indices:n};return S.runKernel(au,s)}const Ip=v({gatherND_:vw});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Sw(e,t){if(t==null)return e.shape.slice();if(Pt(e.shape,t))return t;if(e.shape.length===t.length){const n=[];for(let r=0;r<e.shape.length;r++)t[r]==null&&e.shape[r]!=null?n.push(e.shape[r]):n.push(t[r]);return n}return t}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Tw(e,t,n,r){const s=m(e,"x","dropout");if(y(s.dtype==="float32",()=>`x has to be a floating point tensor since it's going to be scaled, but got a ${s.dtype} tensor instead.`),y(t>=0&&t<1,()=>`rate must be a float in the range [0, 1), but got ${t}.`),t===0)return e instanceof et?s.clone():s;const a=Sw(s,n),o=1-t,i=Y(aa(z(Ir(a,0,1,"float32",r),o)),o);return B(s,i)}const xp=v({dropout_:Tw});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Fa(e){return Math.floor(Math.pow(2,Math.ceil(Math.log(e)/Math.log(2))))}function Dr(e,t,n){const r=1-e%2,s=new Float32Array(e);for(let a=0;a<e;++a){const o=2*Math.PI*a/(e+r-1);s[a]=t-n*Math.cos(o)}return kt(s,"float32")}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function Ew(e,t,n=1){const r=m(e,"predictions","inTopK"),s=m(t,"targets","inTopK");y(r.rank>1,()=>`inTopK() expects the predictions to be of rank 2 or higher, but got ${r.rank}`),y(r.rank-1===s.rank,()=>`predictions rank should be 1 larger than targets rank, but got predictions rank ${r.rank} and targets rank ${s.rank}`),gt(r.shape.slice(0,r.shape.length-1),s.shape,"predictions's shape should be align with the targets' shape, except the last dimension.");const a=r.shape[r.shape.length-1];y(n>0&&n<=a,()=>`'k' passed to inTopK() must be > 0 && <= the predictions last dimension (${a}), but got ${n}`);const o=await r.data(),i=await s.data(),[u,l]=[o.length/a,a],h=Go("bool",u);for(let c=0;c<u;c++){const p=c*l,d=o.subarray(p,p+l),g=[];for(let N=0;N<d.length;N++)g.push({value:d[N],index:N});g.sort((N,w)=>w.value-N.value),h[c]=0;for(let N=0;N<n;N++)if(g[N].index===i[c]){h[c]=1;break}}return e!==r&&r.dispose(),t!==s&&s.dispose(),xt(h,s.shape,"bool")}const Ap=Ew;/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $w(e,t,n,r,s,a="NHWC",o){let i=e;e.rank===3&&(i=O(e,[1,e.shape[0],e.shape[1],e.shape[2]]));let u=t;u.rank===3&&(u=O(t,[1,t.shape[0],t.shape[1],t.shape[2]])),y(i.rank===4,()=>`Error in conv2dDerFilter: input must be rank 4, but got shape ${i.shape}.`),y(u.rank===4,()=>`Error in conv2dDerFilter: dy must be rank 4, but got shape ${u.shape}.`),y(n.length===4,()=>`Error in conv2dDerFilter: filterShape must be length 4, but got ${n}.`);const l=a==="NHWC"?i.shape[3]:i.shape[1],h=a==="NHWC"?u.shape[3]:u.shape[1];y(l===n[2],()=>`Error in conv2dDerFilter: depth of input ${l}) must match input depth in filter (${n[2]}.`),y(h===n[3],()=>`Error in conv2dDerFilter: depth of dy (${h}) must match output depth for filter (${n[3]}).`),Ot("conv2dDerFilter",s,o);const c={x:i,dy:u},p={strides:r,pad:s,dataFormat:a,dimRoundingMode:o,filterShape:n};return S.runKernel(Ii,c,p)}const _w=v({conv2DBackpropFilter_:$w});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Fr(e,t,n){if(n==null||n==="linear")return e;if(n==="relu")return B(e,xa(t));throw new Error(`Cannot compute gradient for fused activation ${n}.`)}function Rr(e,t){let n=t;const r=ea(e.shape,t.shape);return r.length>0&&(n=Q(n,r)),O(n,e.shape)}function Br(e,t,n,r){if(t==="linear")return e;if(t==="relu")return Ln(e);if(t==="elu")return na(e);if(t==="relu6")return $a(e);if(t==="prelu")return wa(e,n);if(t==="leakyrelu")return ua(e,r);if(t==="sigmoid")return Qt(e);throw new Error(`Unknown fused activation ${t}.`)}const Pr=(e,t)=>!(e>0)||t==="linear";/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function kw({x:e,filter:t,strides:n,pad:r,dataFormat:s="NHWC",dilations:a=[1,1],dimRoundingMode:o,bias:i,activation:u="linear",preluActivationWeights:l,leakyreluAlpha:h}){if(u=u||"linear",Pr(S.state.gradientDepth,u)===!1){y(s==="NHWC",()=>`Error in fused conv2d: got dataFormat of ${s} but only NHWC is currently supported for the case of gradient depth is 0 and the activation is not linear.`);let A=Fn(e,t,n,r,s,a,o);return i!=null&&(A=z(A,i)),Br(A,u,l,h)}const c=m(e,"x","conv2d","float32"),p=m(t,"filter","conv2d","float32");let d=c,g=!1;c.rank===3&&(g=!0,d=O(c,[1,c.shape[0],c.shape[1],c.shape[2]])),y(d.rank===4,()=>`Error in fused conv2d: input must be rank 4, but got rank ${d.rank}.`),y(p.rank===4,()=>`Error in fused conv2d: filter must be rank 4, but got rank ${p.rank}.`),Ot("fused conv2d",r,o);const N=s==="NHWC"?d.shape[3]:d.shape[1];y(p.shape[2]===N,()=>`Error in conv2d: depth of input (${N}) must match input depth for filter ${p.shape[2]}.`),y(ne(n,a),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`);const w=On(d.shape,p.shape,n,a,r,o);let T;i!=null&&(T=m(i,"bias","fused conv2d"),[T]=st(T,c),s==="NHWC"?at(w.outShape,T.shape):(y(T.shape.length<=1,()=>`Error in fused conv2d: only supports scalar or 1-D Tensor bias for NCHW format but got the bias of rank-${T.shape.length}.`),y(T.shape.length===0||T.shape[0]===w.outChannels||T.shape[0]===1,()=>`Error in fused conv2d: bias shape (${T.shape}) is not compatible with the number of output channels (${w.outChannels})`)));let x;if(l!=null){const A=l.shape;if(y(A.length<=1||A.length===3,()=>`Error in fused conv2d: only supports scalar, 1-D Tensor or 3-D Tensor PReLU activation weights but got a tensor of rank-${A.length}.`),A.length===1)y(A[0]===1||A[0]===w.outChannels,()=>`Error in fused conv2d: PReLU activation weights (${A}) is not compatible with the number of output channels (${w.outChannels}).`);else if(A.length===3)try{at(A,w.outShape)}catch{const R=`Error in fused conv2d: PReLU activation weights (${A}) is not compatible with the output shape of the conv2d (${w.outShape}).`;throw Error(R)}x=m(l,"prelu weights","fused conv2d")}const $=(A,F)=>{y(s==="NHWC",()=>`Error in gradient of fused conv2D: got dataFormat of ${s} but only NHWC is currently supported.`);const[R,k,_,b]=F,D=Fr(A,_,u);y(wn(a),()=>`Error in gradient of fused conv2D: dilation rates greater than 1 are not yet supported in gradients. Got dilations '${a}'`);const P=jc(k.shape,D,R,n,r),C=_w(k,D,R.shape,n,r),L=[P,C];if(b!=null){const q=Rr(b,D);L.push(q)}return L},E={x:d,filter:p,bias:T,preluActivationWeights:x},I={strides:n,pad:r,dataFormat:s,dilations:a,dimRoundingMode:o,activation:u,leakyreluAlpha:h};return i==null?jt((F,R,k)=>{let _=S.runKernel(ts,E,I);return k([R,F,_]),g&&(_=O(_,[_.shape[1],_.shape[2],_.shape[3]])),{value:_,gradFunc:$}})(d,p):jt((F,R,k,_)=>{let b=S.runKernel(ts,E,I);return _([R,F,b,k]),g&&(b=O(b,[b.shape[1],b.shape[2],b.shape[3]])),{value:b,gradFunc:$}})(d,p,T)}const Iw=v({fusedConv2d_:kw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xw(e,t,n,r,s,a=[1,1],o){let i=e;e.rank===3&&(i=O(e,[1,e.shape[0],e.shape[1],e.shape[2]]));let u=t;u.rank===3&&(u=O(t,[1,t.shape[0],t.shape[1],t.shape[2]]));const l={x:i,dy:u},h={strides:r,pad:s,dimRoundingMode:o,dilations:a,filterShape:n};return S.runKernel(Mi,l,h)}const Aw=v({depthwiseConv2dNativeBackpropFilter_:xw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ow(e,t,n,r,s,a=[1,1],o){let i=t,u=!1;t.rank===3&&(u=!0,i=O(t,[1,t.shape[0],t.shape[1],t.shape[2]]));const l={dy:i,filter:n},h={strides:r,pad:s,dimRoundingMode:o,dilations:a,inputShape:e},c=S.runKernel(Vi,l,h);return u?O(c,[c.shape[1],c.shape[2],c.shape[3]]):c}const Dw=v({depthwiseConv2dNativeBackpropInput_:Ow});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Fw({x:e,filter:t,strides:n,pad:r,dataFormat:s="NHWC",dilations:a=[1,1],dimRoundingMode:o,bias:i,activation:u="linear",preluActivationWeights:l,leakyreluAlpha:h}){if(Pr(S.state.gradientDepth,u)===!1){let I=Er(e,t,n,r,s,a,o);return i!=null&&(I=z(I,i)),Br(I,u,l,h)}const c=m(e,"x","depthwiseConv2d","float32"),p=m(t,"filter","depthwiseConv2d","float32");let d=c,g=!1;c.rank===3&&(g=!0,d=O(c,[1,c.shape[0],c.shape[1],c.shape[2]])),y(d.rank===4,()=>`Error in fused depthwiseConv2d: input must be rank 4, but got rank ${d.rank}.`),y(p.rank===4,()=>`Error in fused depthwiseConv2d: filter must be rank 4, but got rank ${p.rank}.`),y(d.shape[3]===p.shape[2],()=>`Error in fused depthwiseConv2d: number of input channels (${d.shape[3]}) must match the inChannels dimension in filter ${p.shape[2]}.`),a==null&&(a=[1,1]),y(ne(n,a),()=>`Error in fused depthwiseConv2d: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`),Ot("fused depthwiseConv2d",r,o);const N=On(d.shape,p.shape,n,a,r,o,!0);let w;i!=null&&(w=m(i,"bias","fused conv2d"),[w]=st(w,c),at(N.outShape,w.shape));let T;l!=null&&(T=m(l,"prelu weights","fused depthwiseConv2d"));const x=(I,A)=>{y(wn(a),()=>`Error in gradient of fused depthwiseConv2d: dilation rates greater than 1 are not yet supported. Got dilations '${a}'`);const[F,R,k,_]=A,b=Fr(I,k,u),D=Dw(R.shape,b,F,n,r,a,o),P=Aw(R,b,F.shape,n,r,a,o);if(_!=null){const C=Rr(w,b);return[D,P,C]}return[D,P]},$={x:d,filter:p,bias:w,preluActivationWeights:T},E={strides:n,pad:r,dataFormat:s,dilations:a,dimRoundingMode:o,activation:u,leakyreluAlpha:h};return i==null?jt((A,F,R)=>{let k=S.runKernel(es,$,E);return R([F,A,k]),g&&(k=O(k,[k.shape[1],k.shape[2],k.shape[3]])),{value:k,gradFunc:x}})(d,p):jt((A,F,R,k)=>{let _=S.runKernel(es,$,E);return k([F,A,_,R]),g&&(_=O(_,[_.shape[1],_.shape[2],_.shape[3]])),{value:_,gradFunc:x}})(d,p,w)}const Rw=v({fusedDepthwiseConv2d_:Fw});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Bw({a:e,b:t,transposeA:n=!1,transposeB:r=!1,bias:s,activation:a="linear",preluActivationWeights:o,leakyreluAlpha:i=.2}){if(Pr(S.state.gradientDepth,a)===!1){let b=H(e,t,n,r);return s!=null&&(b=z(b,s)),Br(b,a,o,i)}let u=m(e,"a","fused matMul"),l=m(t,"b","fused matMul");[u,l]=st(u,l);const h=n?u.shape[u.rank-2]:u.shape[u.rank-1],c=r?l.shape[l.rank-1]:l.shape[l.rank-2],p=n?u.shape[u.rank-1]:u.shape[u.rank-2],d=r?l.shape[l.rank-2]:l.shape[l.rank-1],g=u.shape.slice(0,-2),N=l.shape.slice(0,-2),w=K(g),T=K(N);y(h===c,()=>`Error in fused matMul: inner shapes (${h}) and (${c}) of Tensors with shapes ${u.shape} and ${l.shape} and transposeA=${n} and transposeB=${r} must match.`);const $=at(u.shape.slice(0,-2),l.shape.slice(0,-2)).concat([p,d]),E=n?O(u,[w,h,p]):O(u,[w,p,h]),I=r?O(l,[T,d,c]):O(l,[T,c,d]);let A;s!=null&&(A=m(s,"bias","fused matMul"),[A]=st(A,u),at($,A.shape));let F;o!=null&&(F=m(o,"prelu weights","fused matMul"));const R=(b,D)=>{const[P,C,L,q]=D,G=Fr(O(b,L.shape),L,a);let tt,Z;if(!n&&!r?(tt=H(G,C,!1,!0),Z=H(P,G,!0,!1)):!n&&r?(tt=H(G,C,!1,!1),Z=H(G,P,!0,!1)):n&&!r?(tt=H(C,G,!1,!0),Z=H(P,G,!1,!1)):(tt=H(C,G,!0,!0),Z=H(G,P,!0,!0)),s!=null){const nt=Rr(q,G);return[tt,Z,nt]}else return[tt,Z]},k={a:E,b:I,bias:A,preluActivationWeights:F},_={transposeA:n,transposeB:r,activation:a,leakyreluAlpha:i};return s==null?jt((D,P,C)=>{const L=S.runKernel(Qr,k,_);return C([D,P,L]),{value:O(L,$),gradFunc:R}})(E,I):jt((D,P,C,L)=>{const q=S.runKernel(Qr,k,_);return L([D,P,q,C]),{value:O(q,$),gradFunc:R}})(E,I,A)}const Pw=v({fusedMatMul_:Bw});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Op=Object.freeze(Object.defineProperty({__proto__:null,conv2d:Iw,depthwiseConv2d:Rw,matMul:Pw},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Cw(e){return Dr(e,.54,.46)}const Lw=v({hammingWindow_:Cw});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zw(e){return Dr(e,.5,.5)}const Dp=v({hannWindow_:zw});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Mw(e,t,n,r=!1,s=0){let a=0;const o=[];for(;a+t<=e.size;)o.push(X(e,a,t)),a+=n;if(r)for(;a<e.size;){const i=a+t-e.size,u=ht([X(e,a,t-i),Je([i],s)]);o.push(u),a+=n}return o.length===0?Ue([],[0,t]):O(ht(o),[o.length,t])}const Fp=v({frame_:Mw});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vw(e,t,n,r,s=Dp){r==null&&(r=Fa(t));const a=Fp(e,t,n),o=B(a,s(t));return Ar(o,r)}const Ww=v({stft_:Vw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Uw(e,t,n,r,s="bilinear",a=0){const o=m(e,"image","cropAndResize"),i=m(t,"boxes","cropAndResize","float32"),u=m(n,"boxInd","cropAndResize","int32"),l=i.shape[0];y(o.rank===4,()=>`Error in cropAndResize: image must be rank 4,but got rank ${o.rank}.`),y(i.rank===2&&i.shape[1]===4,()=>`Error in cropAndResize: boxes must be have size [${l},4] but had shape ${i.shape}.`),y(u.rank===1&&u.shape[0]===l,()=>`Error in cropAndResize: boxInd must be have size [${l}] but had shape ${i.shape}.`),y(r.length===2,()=>`Error in cropAndResize: cropSize must be of length 2, but got length ${r.length}.`),y(r[0]>=1&&r[1]>=1,()=>`cropSize must be atleast [1,1], but was ${r}`),y(s==="bilinear"||s==="nearest",()=>`method must be bilinear or nearest, but was ${s}`);const h={image:o,boxes:i,boxInd:u},c={method:s,extrapolationValue:a,cropSize:r};return S.runKernel(Pi,h,c)}const qw=v({cropAndResize_:Uw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hw(e){const t=m(e,"image","flipLeftRight","float32");y(t.rank===4,()=>`Error in flipLeftRight: image must be rank 4,but got rank ${t.rank}.`);const n={image:t};return S.runKernel(tu,n,{})}const jw=v({flipLeftRight_:Hw});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Gw(e){const t=m(e,"image","grayscaleToRGB"),n=t.rank-1,r=t.shape[n];y(t.rank>=2,()=>`Error in grayscaleToRGB: images must be at least rank 2, but got rank ${t.rank}.`),y(r===1,()=>`Error in grayscaleToRGB: last dimension of a grayscale image should be size 1, but got size ${r}.`);const s=new Array(t.rank);return s.fill(1,0,n),s[n]=3,We(t,s)}const Kw=v({grayscaleToRGB_:Gw});/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xw(e){const t=m(e,"image","RGBToGrayscale"),n=t.rank-1,r=t.shape[n];y(t.rank>=2,()=>`Error in RGBToGrayscale: images must be at least rank 2, but got rank ${t.rank}.`),y(r===3,()=>`Error in RGBToGrayscale: last dimension of an RGB image should be size 3, but got size ${r}.`);const s=t.dtype,a=j(t,"float32"),o=kt([.2989,.587,.114]);let i;switch(t.rank){case 2:i=Te("ij,j->i",a,o);break;case 3:i=Te("ijk,k->ij",a,o);break;case 4:i=Te("ijkl,l->ijk",a,o);break;case 5:i=Te("ijklm,m->ijkl",a,o);break;case 6:i=Te("ijklmn,n->ijklm",a,o);break;default:throw new Error("Not a valid tensor rank.")}return i=_t(i,-1),j(i,s)}const Yw=v({rgbToGrayscale_:Xw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Zw(e,t,n=0,r=.5){const s=m(e,"image","rotateWithOffset","float32");y(s.rank===4,()=>`Error in rotateWithOffset: image must be rank 4,but got rank ${s.rank}.`);const a={image:s},o={radians:t,fillValue:n,center:r};return S.runKernel(ql,a,o)}const Jw=v({rotateWithOffset_:Zw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function en(e,t,n,r,s,a){r==null&&(r=.5),s==null&&(s=Number.NEGATIVE_INFINITY),a==null&&(a=0);const o=e.shape[0];return n=Math.min(n,o),y(0<=r&&r<=1,()=>`iouThreshold must be in [0, 1], but was '${r}'`),y(e.rank===2,()=>`boxes must be a 2D tensor, but was of rank '${e.rank}'`),y(e.shape[1]===4,()=>`boxes must have 4 columns, but 2nd dimension was ${e.shape[1]}`),y(t.rank===1,()=>"scores must be a 1D tensor"),y(t.shape[0]===o,()=>`scores has incompatible shape with boxes. Expected ${o}, but was ${t.shape[0]}`),y(0<=a&&a<=1,()=>`softNmsSigma must be in [0, 1], but was '${a}'`),{maxOutputSize:n,iouThreshold:r,scoreThreshold:s,softNmsSigma:a}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qw(e,t,n,r=.5,s=Number.NEGATIVE_INFINITY){const a=m(e,"boxes","nonMaxSuppression","float32"),o=m(t,"scores","nonMaxSuppression","float32"),i=en(a,o,n,r,s);n=i.maxOutputSize,r=i.iouThreshold,s=i.scoreThreshold;const u={maxOutputSize:n,iouThreshold:r,scoreThreshold:s};return S.runKernel(Cu,{boxes:a,scores:o},u)}const t1=v({nonMaxSuppression_:Qw});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function e1(e,t,n){const r=n1(e,t,n),s=r<0?-(r+1):r;e.splice(s,0,t)}function n1(e,t,n){return s1(e,t,n||r1)}function r1(e,t){return e>t?1:e<t?-1:0}function s1(e,t,n){let r=0,s=e.length,a=0,o=!1;for(;r<s;){a=r+(s-r>>>1);const i=n(t,e[a]);i>0?r=a+1:(s=a,o=!i)}return o?r:-r-1}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Rp(e,t,n,r,s){return Ra(e,t,n,r,s,0)}function Bp(e,t,n,r,s,a){return Ra(e,t,n,r,s,0,!1,a,!0)}function Pp(e,t,n,r,s,a){return Ra(e,t,n,r,s,a,!0)}function Ra(e,t,n,r,s,a,o=!1,i=!1,u=!1){const l=[];for(let w=0;w<t.length;w++)t[w]>s&&l.push({score:t[w],boxIndex:w,suppressBeginIndex:0});l.sort(co);const h=a>0?-.5/a:0,c=[],p=[];for(;c.length<n&&l.length>0;){const w=l.pop(),{score:T,boxIndex:x,suppressBeginIndex:$}=w;if(T<s)break;let E=!1;for(let I=c.length-1;I>=$;--I){const A=a1(e,x,c[I]);if(A>=r){E=!0;break}if(w.score=w.score*o1(r,h,A),w.score<=s)break}w.suppressBeginIndex=c.length,E||(w.score===T?(c.push(x),p.push(w.score)):w.score>s&&e1(l,w,co))}const d=c.length,g=n-d;i&&g>0&&(c.push(...new Array(g).fill(0)),p.push(...new Array(g).fill(0)));const N={selectedIndices:c};return o&&(N.selectedScores=p),u&&(N.validOutputs=d),N}function a1(e,t,n){const r=e.subarray(t*4,t*4+4),s=e.subarray(n*4,n*4+4),a=Math.min(r[0],r[2]),o=Math.min(r[1],r[3]),i=Math.max(r[0],r[2]),u=Math.max(r[1],r[3]),l=Math.min(s[0],s[2]),h=Math.min(s[1],s[3]),c=Math.max(s[0],s[2]),p=Math.max(s[1],s[3]),d=(i-a)*(u-o),g=(c-l)*(p-h);if(d<=0||g<=0)return 0;const N=Math.max(a,l),w=Math.max(o,h),T=Math.min(i,c),x=Math.min(u,p),$=Math.max(T-N,0)*Math.max(x-w,0);return $/(d+g-$)}function o1(e,t,n){const r=Math.exp(t*n*n);return n<=e?r:0}function co(e,t){return e.score-t.score||e.score===t.score&&t.boxIndex-e.boxIndex}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function i1(e,t,n,r=.5,s=Number.NEGATIVE_INFINITY){const a=m(e,"boxes","nonMaxSuppressionAsync"),o=m(t,"scores","nonMaxSuppressionAsync"),i=en(a,o,n,r,s);n=i.maxOutputSize,r=i.iouThreshold,s=i.scoreThreshold;const u=await Promise.all([a.data(),o.data()]),l=u[0],h=u[1],{selectedIndices:c}=Rp(l,h,n,r,s);return a!==e&&a.dispose(),o!==t&&o.dispose(),kt(c,"int32")}const u1=i1;/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function l1(e,t,n,r=.5,s=Number.NEGATIVE_INFINITY,a=0){const o=m(e,"boxes","nonMaxSuppression"),i=m(t,"scores","nonMaxSuppression"),u=en(o,i,n,r,s,a);n=u.maxOutputSize,r=u.iouThreshold,s=u.scoreThreshold,a=u.softNmsSigma;const l={boxes:o,scores:i},h={maxOutputSize:n,iouThreshold:r,scoreThreshold:s,softNmsSigma:a},c=S.runKernel(zu,l,h);return{selectedIndices:c[0],selectedScores:c[1]}}const c1=v({nonMaxSuppressionWithScore_:l1});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function h1(e,t,n,r=.5,s=Number.NEGATIVE_INFINITY,a=0){const o=m(e,"boxes","nonMaxSuppressionAsync"),i=m(t,"scores","nonMaxSuppressionAsync"),u=en(o,i,n,r,s,a);n=u.maxOutputSize,r=u.iouThreshold,s=u.scoreThreshold,a=u.softNmsSigma;const l=await Promise.all([o.data(),i.data()]),h=l[0],c=l[1],{selectedIndices:p,selectedScores:d}=Pp(h,c,n,r,s,a);return o!==e&&o.dispose(),i!==t&&i.dispose(),{selectedIndices:kt(p,"int32"),selectedScores:kt(d)}}const p1=h1;/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function f1(e,t,n,r=.5,s=Number.NEGATIVE_INFINITY,a=!1){const o=m(e,"boxes","nonMaxSuppression"),i=m(t,"scores","nonMaxSuppression"),u=en(o,i,n,r,s,null),l=u.maxOutputSize,h=u.iouThreshold,c=u.scoreThreshold,p={boxes:o,scores:i},d={maxOutputSize:l,iouThreshold:h,scoreThreshold:c,padToMaxOutputSize:a},g=S.runKernel(Lu,p,d);return{selectedIndices:g[0],validOutputs:g[1]}}const d1=v({nonMaxSuppressionPadded_:f1});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function m1(e,t,n,r=.5,s=Number.NEGATIVE_INFINITY,a=!1){const o=m(e,"boxes","nonMaxSuppressionAsync"),i=m(t,"scores","nonMaxSuppressionAsync"),u=en(o,i,n,r,s,null),l=u.maxOutputSize,h=u.iouThreshold,c=u.scoreThreshold,[p,d]=await Promise.all([o.data(),i.data()]),{selectedIndices:g,validOutputs:N}=Bp(p,d,l,h,c,a);return o!==e&&o.dispose(),i!==t&&i.dispose(),{selectedIndices:kt(g,"int32"),validOutputs:U(N,"int32")}}const g1=m1;/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function y1(e,t,n=!1,r=!1){const s=m(e,"images","resizeBilinear");y(s.rank===3||s.rank===4,()=>`Error in resizeBilinear: x must be rank 3 or 4, but got rank ${s.rank}.`),y(t.length===2,()=>`Error in resizeBilinear: new shape must 2D, but got shape ${t}.`),y(r===!1||n===!1,()=>"Error in resizeBilinear: If halfPixelCenters is true, alignCorners must be false.");let a=s,o=!1;s.rank===3&&(o=!0,a=O(s,[1,s.shape[0],s.shape[1],s.shape[2]]));const i={images:a},u={alignCorners:n,halfPixelCenters:r,size:t},l=S.runKernel(nl,i,u);return o?O(l,[l.shape[1],l.shape[2],l.shape[3]]):l}const b1=v({resizeBilinear_:y1});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function w1(e,t,n=!1,r=!1){const s=m(e,"images","resizeNearestNeighbor");y(s.rank===3||s.rank===4,()=>`Error in resizeNearestNeighbor: x must be rank 3 or 4, but got rank ${s.rank}.`),y(t.length===2,()=>`Error in resizeNearestNeighbor: new shape must 2D, but got shape ${t}.`),y(s.dtype==="float32"||s.dtype==="int32",()=>"`images` must have `int32` or `float32` as dtype"),y(r===!1||n===!1,()=>"Error in resizeNearestNeighbor: If halfPixelCenters is true, alignCorners must be false.");let a=s,o=!1;s.rank===3&&(o=!0,a=O(s,[1,s.shape[0],s.shape[1],s.shape[2]]));const i={images:a},u={alignCorners:n,halfPixelCenters:r,size:t},l=S.runKernel(el,i,u);return o?O(l,[l.shape[1],l.shape[2],l.shape[3]]):l}const N1=v({resizeNearestNeighbor_:w1});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function v1(e,t="binary",n=!1,r=.5){const s=m(e,"image","threshold"),a=.2989,o=.587,i=.114,u=s.shape[0]*s.shape[1];let l=B(kt([r]),255),h,c,p,d;if(y(s.rank===3,()=>`Error in threshold: image must be rank 3,but got rank ${s.rank}.`),y(s.shape[2]===3||s.shape[2]===1,()=>`Error in threshold: image color channel must be equal to 3 or 1but got ${s.shape[2]}.`),y(s.dtype==="int32"||s.dtype==="float32",()=>`Error in dtype: image dtype must be int32 or float32,but got dtype ${s.dtype}.`),y(t==="otsu"||t==="binary",()=>`Method must be binary or otsu, but was ${t}`),s.shape[2]===3){[h,c,p]=Ye(s,[1,1,1],-1);const w=B(h,a),T=B(c,o),x=B(p,i);d=z(z(w,T),x)}else d=e;if(t==="otsu"){const w=ta(j(_a(d),"int32"),xt([]),256);l=S1(w,u)}const g=n?$r(d,l):Qe(d,l);return j(B(g,255),"int32")}function S1(e,t){let n=kt([-1]),r=kt([0]),s=kt([0]),a,o,i,u,l,h;for(let c=0;c<e.size-1;c++){a=X(e,0,c+1),o=X(e,c+1),l=Y(Q(a),t),h=Y(Q(o),t);const p=Q(B(a,me(0,a.size)));i=Y(p,Q(a));const d=Je(o.shape,a.size),g=z(me(0,o.size),d),N=B(o,g);u=Y(Q(N),Q(o));const w=W(i,u),T=W(i,u),x=B(l,h);s=B(B(x,w),T);const $=Qe(s,r);r=te($,s,r),n=te($,kt([c]),n)}return n}const T1=v({threshold_:v1});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function E1(e,t,n="nearest",r="constant",s=0,a){const o=m(e,"image","transform","float32"),i=m(t,"transforms","transform","float32");y(o.rank===4,()=>`Error in transform: image must be rank 4,but got rank ${o.rank}.`),y(i.rank===2&&(i.shape[0]===o.shape[0]||i.shape[0]===1)&&i.shape[1]===8,()=>"Error in transform: Input transform should be batch x 8 or 1 x 8"),y(a==null||a.length===2,()=>`Error in transform: outputShape must be [height, width] or null, but got ${a}.`);const u={image:o,transforms:i},l={interpolation:n,fillMode:r,fillValue:s,outputShape:a};return S.runKernel(Ll,u,l)}const $1=v({transform_:E1});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _1(e,t,n){const r=m(e,"a","bandPart");y(r.rank>=2,()=>`bandPart(): Rank must be at least 2, got ${r.rank}.`);const s=r.shape,[a,o]=r.shape.slice(-2);let i,u;typeof t=="number"?(y(t%1===0,()=>`bandPart(): numLower must be an integer, got ${t}.`),y(t<=a,()=>`bandPart(): numLower (${t}) must not be greater than the number of rows (${a}).`),i=m(t<0?a:t,"numLower","bandPart")):(y(t.dtype==="int32",()=>"bandPart(): numLower's dtype must be an int32."),i=te(yr(t,0),a,Sn(t,a))),typeof n=="number"?(y(n%1===0,()=>`bandPart(): numUpper must be an integer, got ${n}.`),y(n<=o,()=>`bandPart(): numUpper (${n}) must not be greater than the number of columns (${o}).`),u=m(n<0?o:n,"numUpper","bandPart")):(y(n.dtype==="int32",()=>"bandPart(): numUpper's dtype must be an int32."),u=te(yr(n,0),o,Sn(n,o)));const l=O(me(0,a,1,"int32"),[-1,1]),h=me(0,o,1,"int32"),c=W(l,h),p=Nn($r(c,i),ia(c,Bt(u))),d=De([a,o],r.dtype);return O(Gt(be(O(r,[-1,a,o])).map(g=>te(p,g,d))),s)}const k1=v({bandPart_:_1});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function I1(e){let t;if(Array.isArray(e)){t=!1,y(e!=null&&e.length>0,()=>"Gram-Schmidt process: input must not be null, undefined, or empty");const s=e[0].shape[0];for(let a=1;a<e.length;++a)y(e[a].shape[0]===s,()=>`Gram-Schmidt: Non-unique lengths found in the input vectors: (${e[a].shape[0]} vs. ${s})`)}else t=!0,e=Ye(e,e.shape[0],0).map(s=>Mt(s,[0]));y(e.length<=e[0].shape[0],()=>`Gram-Schmidt: Number of vectors (${e.length}) exceeds number of dimensions (${e[0].shape[0]}).`);const n=[],r=e;for(let s=0;s<e.length;++s)n.push(S.tidy(()=>{let a=r[s];if(s>0)for(let o=0;o<s;++o){const i=B(Q(B(n[o],a)),n[o]);a=W(a,i)}return Y(a,Pn(a,"euclidean"))}));return t?Gt(n,0):n}const x1=v({gramSchmidt_:I1});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function A1(e,t=!1){if(y(e.rank>=2,()=>`qr() requires input tensor to have a rank >= 2, but got rank ${e.rank}`),e.rank===2)return ho(e,t);{const n=e.shape.slice(0,e.shape.length-2).reduce((u,l)=>u*l),r=be(O(e,[n,e.shape[e.shape.length-2],e.shape[e.shape.length-1]]),0),s=[],a=[];r.forEach(u=>{const[l,h]=ho(u,t);s.push(l),a.push(h)});const o=O(Gt(s,0),e.shape),i=O(Gt(a,0),e.shape);return[o,i]}}function ho(e,t=!1){return S.tidy(()=>{y(e.shape.length===2,()=>`qr2d() requires a 2D Tensor, but got a ${e.shape.length}D Tensor.`);const n=e.shape[0],r=e.shape[1];let s=sa(n),a=Jt(e);const o=Ue([[1]],[1,1]);let i=Jt(o);const u=n>=r?r:n;for(let l=0;l<u;++l){const h=a,c=i,p=s;[i,a,s]=S.tidy(()=>{const d=X(a,[l,l],[n-l,1]),g=Pn(d),N=X(a,[l,l],[1,1]),w=te(Qe(N,0),Ue([[-1]]),Ue([[1]])),T=W(N,B(w,g)),x=Y(d,T);x.shape[0]===1?i=Jt(o):i=ht([o,X(x,[1,0],[x.shape[0]-1,x.shape[1]])],0);const $=Bt(Y(H(w,T),g)),E=X(a,[l,0],[n-l,r]),I=B($,i),A=$n(i);if(l===0)a=W(E,H(I,H(A,E)));else{const k=W(E,H(I,H(A,E)));a=ht([X(a,[0,0],[l,r]),k],0)}const F=$n(I),R=X(s,[0,l],[n,s.shape[1]-l]);if(l===0)s=W(R,H(H(R,i),F));else{const k=W(R,H(H(R,i),F));s=ht([X(s,[0,0],[n,l]),k],1)}return[i,a,s]}),mt([h,c,p])}return!t&&n>r&&(s=X(s,[0,0],[n,r]),a=X(a,[0,0],[r,r])),[s,a]})}const O1=v({qr_:A1});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var dt;(function(e){e[e.NONE=0]="NONE",e[e.MEAN=1]="MEAN",e[e.SUM=2]="SUM",e[e.SUM_BY_NONZERO_WEIGHTS=3]="SUM_BY_NONZERO_WEIGHTS"})(dt||(dt={}));function D1(e,t,n=dt.SUM_BY_NONZERO_WEIGHTS){const r=m(e,"losses","computeWeightedLoss");let s=null;t!=null&&(s=m(t,"weights","computeWeightedLoss"));const a=s==null?r:B(r,s);if(n===dt.NONE)return a;if(n===dt.SUM)return Q(a);if(n===dt.MEAN){if(s==null)return vn(a);{const o=r.size/s.size,i=Y(Q(a),Q(s));return o>1?Y(i,U(o)):i}}if(n===dt.SUM_BY_NONZERO_WEIGHTS){if(s==null)return Y(Q(a),U(r.size));{const o=B(s,ue(r.shape)),i=j(Q(ga(o,U(0))),"float32");return Y(Q(a),i)}}throw Error(`Unknown reduction: ${n}`)}const re=v({computeWeightedLoss_:D1});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function F1(e,t,n,r=dt.SUM_BY_NONZERO_WEIGHTS){const s=m(e,"labels","absoluteDifference"),a=m(t,"predictions","absoluteDifference");let o=null;n!=null&&(o=m(n,"weights","absoluteDifference")),gt(s.shape,a.shape,"Error in absoluteDifference: ");const i=Tt(W(s,a));return re(i,o,r)}const R1=v({absoluteDifference_:F1});function B1(e,t,n,r,s=dt.SUM_BY_NONZERO_WEIGHTS){const a=m(e,"labels","cosineDistance"),o=m(t,"predictions","cosineDistance");let i=null;r!=null&&(i=m(r,"weights","cosineDistance")),gt(a.shape,o.shape,"Error in cosineDistance: ");const u=U(1),l=W(u,Q(B(a,o),n,!0));return re(l,i,s)}const P1=v({cosineDistance_:B1});function C1(e,t,n,r=dt.SUM_BY_NONZERO_WEIGHTS){let s=m(e,"labels","hingeLoss");const a=m(t,"predictions","hingeLoss");let o=null;n!=null&&(o=m(n,"weights","hingeLoss")),gt(s.shape,a.shape,"Error in hingeLoss: ");const i=U(1);s=W(B(U(2),s),i);const u=Ln(W(i,B(s,a)));return re(u,o,r)}const L1=v({hingeLoss_:C1});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function z1(e,t,n,r=1,s=dt.SUM_BY_NONZERO_WEIGHTS){const a=m(e,"labels","huberLoss"),o=m(t,"predictions","huberLoss");let i=null;n!=null&&(i=m(n,"weights","huberLoss")),gt(a.shape,o.shape,"Error in huberLoss: ");const u=U(r),l=Tt(W(o,a)),h=Sn(l,u),c=W(l,h),p=z(B(U(.5),At(h)),B(u,c));return re(p,i,s)}const M1=v({huberLoss_:z1});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function V1(e,t,n,r=1e-7,s=dt.SUM_BY_NONZERO_WEIGHTS){const a=m(e,"labels","logLoss"),o=m(t,"predictions","logLoss");let i=null;n!=null&&(i=m(n,"weights","logLoss")),gt(a.shape,o.shape,"Error in logLoss: ");const u=U(1),l=U(r),h=Bt(B(a,Ke(z(o,l)))),c=B(W(u,a),Ke(z(W(u,o),l))),p=W(h,c);return re(p,i,s)}const W1=v({logLoss_:V1});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function U1(e,t,n,r=dt.SUM_BY_NONZERO_WEIGHTS){const s=m(e,"labels","meanSquaredError"),a=m(t,"predictions","meanSquaredError");let o=null;n!=null&&(o=m(n,"weights","meanSquaredError")),gt(s.shape,a.shape,"Error in meanSquaredError: ");const i=Ia(s,a);return re(i,o,r)}const q1=v({meanSquaredError_:U1});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function H1(e,t){const n=m(e,"labels","sigmoidCrossEntropyWithLogits"),r=m(t,"logits","sigmoidCrossEntropyWithLogits");gt(n.shape,r.shape,"Error in sigmoidCrossEntropyWithLogits: ");const s=Ln(r),a=B(r,n),o=la(de(Bt(Tt(r))));return z(W(s,a),o)}function j1(e,t,n,r=0,s=dt.SUM_BY_NONZERO_WEIGHTS){let a=m(e,"multiClassLabels","sigmoidCrossEntropy");const o=m(t,"logits","sigmoidCrossEntropy");let i=null;if(n!=null&&(i=m(n,"weights","sigmoidCrossEntropy")),gt(a.shape,o.shape,"Error in sigmoidCrossEntropy: "),r>0){const l=U(r),h=U(1),c=U(.5);a=z(B(a,W(h,l)),B(c,l))}const u=H1(a,o);return re(u,i,s)}const G1=v({sigmoidCrossEntropy_:j1});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function K1(e,t,n=-1){if(n===-1&&(n=t.rank-1),n!==t.rank-1)throw Error(`Softmax cross entropy along a non-last dimension is not yet supported. Labels / logits was rank ${t.rank} and dim was ${n}`);return jt((s,a,o)=>{const u=ha(a,[n],!0),l=W(j(a,"float32"),u);o([s,l]);const h=Bt(B(l,s));return{value:Q(h,[n]),gradFunc:(d,g)=>{const[N,w]=g,T=Bn(d.shape,[n]);return[B(O(d,T),W(j(N,"float32"),de(w))),B(O(d,T),W(de(w),j(N,"float32")))]}}})(e,t)}function X1(e,t,n,r=0,s=dt.SUM_BY_NONZERO_WEIGHTS){let a=m(e,"onehotLabels","softmaxCrossEntropy");const o=m(t,"logits","softmaxCrossEntropy");let i=null;if(n!=null&&(i=m(n,"weights","softmaxCrossEntropy")),gt(a.shape,o.shape,"Error in softmaxCrossEntropy: "),r>0){const l=U(r),h=U(1),c=U(a.shape[1]);a=z(B(a,W(h,l)),Y(l,c))}const u=K1(a,o);return re(u,i,s)}const Y1=v({softmaxCrossEntropy_:X1});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Z1(e,t,n,r){const s=m(e,"indices","sparseFillEmptyRows","int32"),a=m(t,"values","sparseFillEmptyRows"),o=m(n,"denseShape","sparseFillEmptyRows","int32"),i=m(r,"defaultValue","sparseFillEmptyRows",a.dtype);if(s.rank!==2)throw new Error(`Indices should be Tensor2D but received shape
        ${s.shape}`);if(a.rank!==1)throw new Error(`Values should be Tensor1D but received shape ${a.shape}`);if(o.rank!==1)throw new Error(`Dense shape should be Tensor1D but received shape ${o.shape}`);if(i.rank!==0)throw new Error(`Default value should be a scalar but received shape ${i.shape}`);const u={indices:s,values:a,denseShape:o,defaultValue:i},l=S.runKernel(Tl,u);return{outputIndices:l[0],outputValues:l[1],emptyRowIndicator:l[2],reverseIndexMap:l[3]}}const J1=v({sparseFillEmptyRows_:Z1});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Q1(e,t,n){const r=m(e,"inputIndices","sparseReshape","int32"),s=m(t,"inputShape","sparseReshape","int32"),a=m(n,"newShape","sparseReshape","int32");if(r.rank!==2)throw new Error(`Input indices should be Tensor2D but received shape
        ${r.shape}`);if(s.rank!==1)throw new Error(`Input shape should be Tensor1D but received shape ${s.shape}`);if(a.rank!==1)throw new Error(`New shape should be Tensor1D but received shape ${a.shape}`);const o={inputIndices:r,inputShape:s,newShape:a},i=S.runKernel(El,o);return{outputIndices:i[0],outputShape:i[1]}}const tN=v({sparseReshape_:Q1});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function eN(e,t,n){const r=m(e,"data","sparseSegmentMean"),s=m(t,"indices","sparseSegmentMean","int32"),a=m(n,"segmentIds","sparseSegmentMean","int32");if(r.rank<1)throw new Error("Data should be at least 1 dimensional but received scalar");if(s.rank!==1)throw new Error(`Indices should be Tensor1D but received shape
          ${s.shape}`);if(a.rank!==1)throw new Error(`Segment ids should be Tensor1D but received shape
          ${a.shape}`);const o={data:r,indices:s,segmentIds:a};return S.runKernel($l,o)}const nN=v({sparseSegmentMean_:eN});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function rN(e,t,n){const r=m(e,"data","sparseSegmentSum"),s=m(t,"indices","sparseSegmentSum","int32"),a=m(n,"segmentIds","sparseSegmentSum","int32");if(r.rank<1)throw new Error("Data should be at least 1 dimensional but received scalar");if(s.rank!==1)throw new Error(`Indices should be Tensor1D but received shape
         ${s.shape}`);if(a.rank!==1)throw new Error(`Segment ids should be Tensor1D but received shape
         ${a.shape}`);const o={data:r,indices:s,segmentIds:a};return S.runKernel(_l,o)}const sN=v({sparseSegmentSum_:rN});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function aN(e,t,n,r,s,a,o,i){const u=m(e,"data","stringNGrams","string");if(u.dtype!=="string")throw new Error("Data must be of datatype string");if(u.shape.length!==1)throw new Error(`Data must be a vector, saw: ${u.shape}`);const l=m(t,"dataSplits","stringNGrams");if(l.dtype!=="int32")throw new Error("Data splits must be of datatype int32");const h={separator:n,nGramWidths:r,leftPad:s,rightPad:a,padWidth:o,preserveShortSequences:i},c={data:u,dataSplits:l},p=S.runKernel(Ol,c,h);return{nGrams:p[0],nGramsSplits:p[1]}}const oN=v({stringNGrams_:aN});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function iN(e,t,n=!0){const r=m(e,"input","stringSplit","string"),s=m(t,"delimiter","stringSplit","string");if(r.rank!==1)throw new Error(`Input should be Tensor1D but received shape ${r.shape}`);if(s.rank!==0)throw new Error(`Delimiter should be a scalar but received shape ${s.shape}`);const a={skipEmpty:n},o={input:r,delimiter:s},i=S.runKernel(Dl,o,a);return{indices:i[0],values:i[1],shape:i[2]}}const uN=v({stringSplit_:iN});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function lN(e,t){const n=m(e,"input","stringToHashBucketFast","string"),r={numBuckets:t};if(t<=0)throw new Error("Number of buckets must be at least 1");const s={input:n};return S.runKernel(Fl,s,r)}const cN=v({stringToHashBucketFast_:lN});/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hN(e,t,n,r=!0){const s=m(e,"input","staticRegexReplace","string"),a={pattern:t,rewrite:n,replaceGlobal:r};return S.runKernel(xl,{x:s},a)}const pN=v({staticRegexReplace_:hN});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Cp={fft:xr,ifft:En,rfft:Ar,irfft:ka},Lp={hammingWindow:Lw,hannWindow:Dp,frame:Fp,stft:Ww},zn={flipLeftRight:jw,grayscaleToRGB:Kw,resizeNearestNeighbor:N1,resizeBilinear:b1,rgbToGrayscale:Yw,rotateWithOffset:Jw,cropAndResize:qw,nonMaxSuppression:t1,nonMaxSuppressionAsync:u1,nonMaxSuppressionWithScore:c1,nonMaxSuppressionWithScoreAsync:p1,nonMaxSuppressionPadded:d1,nonMaxSuppressionPaddedAsync:g1,threshold:T1,transform:$1},zp={bandPart:k1,gramSchmidt:x1,qr:O1},Mp={absoluteDifference:R1,computeWeightedLoss:re,cosineDistance:P1,hingeLoss:L1,huberLoss:M1,logLoss:W1,meanSquaredError:q1,sigmoidCrossEntropy:G1,softmaxCrossEntropy:Y1},Vp={sparseFillEmptyRows:J1,sparseReshape:tN,sparseSegmentMean:nN,sparseSegmentSum:sN},Wp={stringNGrams:oN,stringSplit:uN,stringToHashBucketFast:cN,staticRegexReplace:pN};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const fN=new Map,gs=new Map;class Up{getClassName(){return this.constructor.className}static fromConfig(t,n){return new t(n)}}class ae{constructor(){this.classNameMap={}}static getMap(){return ae.instance==null&&(ae.instance=new ae),ae.instance}static register(t){ae.getMap().classNameMap[t.className]=[t,t.fromConfig]}}function qp(e,t,n){y(e.className!=null,()=>"Class being registered does not have the static className property defined."),y(typeof e.className=="string",()=>"className is required to be a string, but got type "+typeof e.className),y(e.className.length>0,()=>"Class being registered has an empty-string as its className, which is disallowed."),typeof t>"u"&&(t="Custom"),typeof n>"u"&&(n=e.className);const r=n,s=t+">"+r;return ae.register(e),fN.set(s,e),gs.set(e,s),e}function dN(e){return gs.has(e)?gs.get(e):e.className}const mN=Object.freeze(Object.defineProperty({__proto__:null,Serializable:Up,SerializationMap:ae,getRegisteredName:dN,registerClass:qp},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class we extends Up{minimize(t,n=!1,r){const{value:s,grads:a}=this.computeGradients(t,r);if(r!=null){const o=r.map(i=>({name:i.name,tensor:a[i.name]}));this.applyGradients(o)}else this.applyGradients(a);return mt(a),n?s:(s.dispose(),null)}get iterations(){return this.iterations_==null&&(this.iterations_=0),this.iterations_}incrementIterations(){this.iterations_=this.iterations+1}computeGradients(t,n){return bh(t,n)}dispose(){this.iterations_!=null&&mt(this.iterations_)}async saveIterations(){return this.iterations_==null&&(this.iterations_=0),{name:"iter",tensor:U(this.iterations_,"int32")}}async getWeights(){throw new Error("getWeights() is not implemented for this optimizer yet.")}async setWeights(t){throw new Error(`setWeights() is not implemented for this optimizer class ${this.getClassName()}`)}async extractIterations(t){return this.iterations_=(await t[0].tensor.data())[0],t.slice(1)}}Object.defineProperty(we,Symbol.hasInstance,{value:e=>e.minimize!=null&&e.computeGradients!=null&&e.applyGradients!=null});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Ba extends we{static get className(){return"Adadelta"}constructor(t,n,r=null){super(),this.learningRate=t,this.rho=n,this.epsilon=r,this.accumulatedGrads=[],this.accumulatedUpdates=[],r==null&&(this.epsilon=S.backend.epsilon())}applyGradients(t){(Array.isArray(t)?t.map(r=>r.name):Object.keys(t)).forEach((r,s)=>{const a=S.registeredVariables[r],o=!1;this.accumulatedGrads[s]==null&&(this.accumulatedGrads[s]={originalName:`${r}/accum_grad`,variable:V(()=>Et(a).variable(o))}),this.accumulatedUpdates[s]==null&&(this.accumulatedUpdates[s]={originalName:`${r}/accum_var`,variable:V(()=>Et(a).variable(o))});const i=Array.isArray(t)?t[s].tensor:t[r];if(i==null)return;const u=this.accumulatedGrads[s].variable,l=this.accumulatedUpdates[s].variable;V(()=>{const h=z(B(u,this.rho),B(At(i),1-this.rho)),c=B(Y(Ht(z(l,this.epsilon)),Ht(z(u,this.epsilon))),i),p=z(B(l,this.rho),B(At(c),1-this.rho));u.assign(h),l.assign(p);const d=z(B(c,-this.learningRate),a);a.assign(d)})}),this.incrementIterations()}dispose(){this.accumulatedUpdates!=null&&(mt(this.accumulatedGrads.map(t=>t.variable)),mt(this.accumulatedUpdates.map(t=>t.variable)))}async getWeights(){const t=[...this.accumulatedGrads,...this.accumulatedUpdates];return[await this.saveIterations()].concat(t.map(n=>({name:n.originalName,tensor:n.variable})))}async setWeights(t){t=await this.extractIterations(t);const n=t.length/2,r=!1;this.accumulatedGrads=t.slice(0,n).map(s=>({originalName:s.name,variable:s.tensor.variable(r)})),this.accumulatedUpdates=t.slice(n,n*2).map(s=>({originalName:s.name,variable:s.tensor.variable(r)}))}getConfig(){return{learningRate:this.learningRate,rho:this.rho,epsilon:this.epsilon}}static fromConfig(t,n){return new t(n.learningRate,n.rho,n.epsilon)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Pa extends we{static get className(){return"Adagrad"}constructor(t,n=.1){super(),this.learningRate=t,this.initialAccumulatorValue=n,this.accumulatedGrads=[]}applyGradients(t){(Array.isArray(t)?t.map(r=>r.name):Object.keys(t)).forEach((r,s)=>{const a=S.registeredVariables[r];this.accumulatedGrads[s]==null&&(this.accumulatedGrads[s]={originalName:`${r}/accumulator`,variable:V(()=>Je(a.shape,this.initialAccumulatorValue).variable(!1))});const o=Array.isArray(t)?t[s].tensor:t[r];if(o==null)return;const i=this.accumulatedGrads[s].variable;V(()=>{const u=z(i,At(o));i.assign(u);const l=z(B(Y(o,Ht(z(u,S.backend.epsilon()))),-this.learningRate),a);a.assign(l)})}),this.incrementIterations()}dispose(){this.accumulatedGrads!=null&&mt(this.accumulatedGrads.map(t=>t.variable))}async getWeights(){return[await this.saveIterations()].concat(this.accumulatedGrads.map(t=>({name:t.originalName,tensor:t.variable})))}async setWeights(t){t=await this.extractIterations(t);const n=!1;this.accumulatedGrads=t.map(r=>({originalName:r.name,variable:r.tensor.variable(n)}))}getConfig(){return{learningRate:this.learningRate,initialAccumulatorValue:this.initialAccumulatorValue}}static fromConfig(t,n){return new t(n.learningRate,n.initialAccumulatorValue)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Ca extends we{static get className(){return"Adam"}constructor(t,n,r,s=null){super(),this.learningRate=t,this.beta1=n,this.beta2=r,this.epsilon=s,this.accumulatedFirstMoment=[],this.accumulatedSecondMoment=[],V(()=>{this.accBeta1=U(n).variable(),this.accBeta2=U(r).variable()}),s==null&&(this.epsilon=S.backend.epsilon())}applyGradients(t){const n=Array.isArray(t)?t.map(r=>r.name):Object.keys(t);V(()=>{const r=W(1,this.accBeta1),s=W(1,this.accBeta2);n.forEach((a,o)=>{const i=S.registeredVariables[a],u=!1;this.accumulatedFirstMoment[o]==null&&(this.accumulatedFirstMoment[o]={originalName:`${a}/m`,variable:V(()=>Et(i).variable(u))}),this.accumulatedSecondMoment[o]==null&&(this.accumulatedSecondMoment[o]={originalName:`${a}/v`,variable:V(()=>Et(i).variable(u))});const l=Array.isArray(t)?t[o].tensor:t[a];if(l==null)return;const h=this.accumulatedFirstMoment[o].variable,c=this.accumulatedSecondMoment[o].variable,p=z(B(h,this.beta1),B(l,1-this.beta1)),d=z(B(c,this.beta2),B(At(l),1-this.beta2)),g=Y(p,r),N=Y(d,s);h.assign(p),c.assign(d);const w=z(B(Y(g,z(Ht(N),this.epsilon)),-this.learningRate),i);i.assign(w)}),this.accBeta1.assign(B(this.accBeta1,this.beta1)),this.accBeta2.assign(B(this.accBeta2,this.beta2))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.accBeta2.dispose(),this.accumulatedFirstMoment!=null&&mt(this.accumulatedFirstMoment.map(t=>t.variable)),this.accumulatedSecondMoment!=null&&mt(this.accumulatedSecondMoment.map(t=>t.variable))}async getWeights(){const t=[...this.accumulatedFirstMoment,...this.accumulatedSecondMoment];return[await this.saveIterations()].concat(t.map(n=>({name:n.originalName,tensor:n.variable})))}async setWeights(t){t=await this.extractIterations(t),V(()=>{this.accBeta1.assign(Ge(this.beta1,this.iterations_+1)),this.accBeta2.assign(Ge(this.beta2,this.iterations_+1))});const n=t.length/2,r=!1;this.accumulatedFirstMoment=t.slice(0,n).map(s=>({originalName:s.name,variable:s.tensor.variable(r)})),this.accumulatedSecondMoment=t.slice(n,n*2).map(s=>({originalName:s.name,variable:s.tensor.variable(r)}))}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon}}static fromConfig(t,n){return new t(n.learningRate,n.beta1,n.beta2,n.epsilon)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class La extends we{static get className(){return"Adamax"}constructor(t,n,r,s=null,a=0){super(),this.learningRate=t,this.beta1=n,this.beta2=r,this.epsilon=s,this.decay=a,this.accumulatedFirstMoment=[],this.accumulatedWeightedInfNorm=[],V(()=>{this.iteration=U(0).variable(),this.accBeta1=U(n).variable()}),s==null&&(this.epsilon=S.backend.epsilon())}applyGradients(t){const n=Array.isArray(t)?t.map(r=>r.name):Object.keys(t);V(()=>{const r=W(1,this.accBeta1),s=Y(-this.learningRate,z(B(this.iteration,this.decay),1));n.forEach((a,o)=>{const i=S.registeredVariables[a],u=!1;this.accumulatedFirstMoment[o]==null&&(this.accumulatedFirstMoment[o]={originalName:`${a}/m`,variable:Et(i).variable(u)}),this.accumulatedWeightedInfNorm[o]==null&&(this.accumulatedWeightedInfNorm[o]={originalName:`${a}/v`,variable:Et(i).variable(u)});const l=Array.isArray(t)?t[o].tensor:t[a];if(l==null)return;const h=this.accumulatedFirstMoment[o].variable,c=this.accumulatedWeightedInfNorm[o].variable,p=z(B(h,this.beta1),B(l,1-this.beta1)),d=B(c,this.beta2),g=Tt(l),N=ma(d,g);h.assign(p),c.assign(N);const w=z(B(Y(s,r),Y(p,z(N,this.epsilon))),i);i.assign(w)}),this.iteration.assign(z(this.iteration,1)),this.accBeta1.assign(B(this.accBeta1,this.beta1))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.iteration.dispose(),this.accumulatedFirstMoment!=null&&mt(this.accumulatedFirstMoment.map(t=>t.variable)),this.accumulatedWeightedInfNorm!=null&&mt(this.accumulatedWeightedInfNorm.map(t=>t.variable))}async getWeights(){throw new Error("getWeights() is not implemented for Adamax yet.")}async setWeights(t){throw new Error("setWeights() is not implemented for Adamax yet.")}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon,decay:this.decay}}static fromConfig(t,n){return new t(n.learningRate,n.beta1,n.beta2,n.epsilon,n.decay)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Cr extends we{static get className(){return"SGD"}constructor(t){super(),this.learningRate=t,this.setLearningRate(t)}applyGradients(t){(Array.isArray(t)?t.map(r=>r.name):Object.keys(t)).forEach((r,s)=>{const a=Array.isArray(t)?t[s].tensor:t[r];if(a==null)return;const o=S.registeredVariables[r];V(()=>{const i=z(B(this.c,a),o);o.assign(i)})}),this.incrementIterations()}setLearningRate(t){this.learningRate=t,this.c!=null&&this.c.dispose(),this.c=Rt(U(-t))}dispose(){this.c.dispose()}async getWeights(){return[await this.saveIterations()]}async setWeights(t){if(t=await this.extractIterations(t),t.length!==0)throw new Error("SGD optimizer does not have settable weights.")}getConfig(){return{learningRate:this.learningRate}}static fromConfig(t,n){return new t(n.learningRate)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class za extends Cr{static get className(){return"Momentum"}constructor(t,n,r=!1){super(t),this.learningRate=t,this.momentum=n,this.useNesterov=r,this.accumulations=[],this.m=U(this.momentum)}applyGradients(t){(Array.isArray(t)?t.map(r=>r.name):Object.keys(t)).forEach((r,s)=>{const a=S.registeredVariables[r];this.accumulations[s]==null&&(this.accumulations[s]={originalName:`${r}/momentum`,variable:V(()=>Et(a).variable(!1))});const o=this.accumulations[s].variable,i=Array.isArray(t)?t[s].tensor:t[r];i!=null&&V(()=>{let u;const l=z(B(this.m,o),i);this.useNesterov?u=z(B(this.c,z(i,B(l,this.m))),a):u=z(B(this.c,l),a),o.assign(l),a.assign(u)})}),this.incrementIterations()}dispose(){this.m.dispose(),this.accumulations!=null&&mt(this.accumulations.map(t=>t.variable))}setMomentum(t){this.momentum=t}async getWeights(){return[await this.saveIterations()].concat(this.accumulations.map(t=>({name:t.originalName,tensor:t.variable})))}async setWeights(t){t=await this.extractIterations(t);const n=!1;this.accumulations=t.map(r=>({originalName:r.name,variable:r.tensor.variable(n)}))}getConfig(){return{learningRate:this.learningRate,momentum:this.momentum,useNesterov:this.useNesterov}}static fromConfig(t,n){return new t(n.learningRate,n.momentum,n.useNesterov)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Ma extends we{static get className(){return"RMSProp"}constructor(t,n=.9,r=0,s=null,a=!1){if(super(),this.learningRate=t,this.decay=n,this.momentum=r,this.epsilon=s,this.accumulatedMeanSquares=[],this.accumulatedMoments=[],this.accumulatedMeanGrads=[],this.centered=a,s==null&&(this.epsilon=S.backend.epsilon()),t==null)throw new Error("learningRate for RMSPropOptimizer must be defined.")}applyGradients(t){(Array.isArray(t)?t.map(r=>r.name):Object.keys(t)).forEach((r,s)=>{const a=S.registeredVariables[r],o=!1;this.accumulatedMeanSquares[s]==null&&(this.accumulatedMeanSquares[s]={originalName:`${r}/rms`,variable:V(()=>Et(a).variable(o))}),this.accumulatedMoments[s]==null&&(this.accumulatedMoments[s]={originalName:`${r}/momentum`,variable:V(()=>Et(a).variable(o))}),this.accumulatedMeanGrads[s]==null&&this.centered&&(this.accumulatedMeanGrads[s]={originalName:`${r}/mg`,variable:V(()=>Et(a).variable(o))});const i=Array.isArray(t)?t[s].tensor:t[r];if(i==null)return;const u=this.accumulatedMeanSquares[s].variable,l=this.accumulatedMoments[s].variable;V(()=>{const h=z(B(u,this.decay),B(At(i),1-this.decay));if(this.centered){const c=this.accumulatedMeanGrads[s].variable,p=z(B(c,this.decay),B(i,1-this.decay)),d=Y(B(i,this.learningRate),Ht(W(h,z(At(p),this.epsilon)))),g=z(B(l,this.momentum),d);u.assign(h),c.assign(p),l.assign(g);const N=W(a,g);a.assign(N)}else{const c=z(B(u,this.decay),B(At(i),1-this.decay)),p=z(B(l,this.momentum),Y(B(i,this.learningRate),Ht(z(c,this.epsilon))));u.assign(c),l.assign(p);const d=W(a,p);a.assign(d)}})}),this.incrementIterations()}dispose(){this.accumulatedMeanSquares!=null&&mt(this.accumulatedMeanSquares.map(t=>t.variable)),this.accumulatedMeanGrads!=null&&this.centered&&mt(this.accumulatedMeanGrads.map(t=>t.variable)),this.accumulatedMoments!=null&&mt(this.accumulatedMoments.map(t=>t.variable))}async getWeights(){const t=[...this.accumulatedMeanSquares,...this.accumulatedMoments];return this.centered&&t.push(...this.accumulatedMeanGrads),[await this.saveIterations()].concat(t.map(n=>({name:n.originalName,tensor:n.variable})))}async setWeights(t){t=await this.extractIterations(t);const n=this.centered?t.length/3:t.length/2,r=!1;this.accumulatedMeanSquares=t.slice(0,n).map(s=>({originalName:s.name,variable:s.tensor.variable(r)})),this.accumulatedMoments=t.slice(n,n*2).map(s=>({originalName:s.name,variable:s.tensor.variable(r)})),this.centered&&(this.accumulatedMeanGrads=t.slice(n*2,n*3).map(s=>({originalName:s.name,variable:s.tensor.variable(r)})))}getConfig(){return{learningRate:this.learningRate,decay:this.decay,momentum:this.momentum,epsilon:this.epsilon,centered:this.centered}}static fromConfig(t,n){return new t(n.learningRate,n.decay,n.momentum,n.epsilon,n.centered)}}/**
 * @license
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const gN=[Ba,Pa,Ca,La,za,Ma,Cr];function yN(){for(const e of gN)qp(e)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const bN="model",wN=".json",NN=".weights.bin";function po(e){return new Promise(t=>setTimeout(t)).then(e)}class Fe{constructor(t){if(!M().getBool("IS_BROWSER"))throw new Error("browserDownloads() cannot proceed because the current environment is not a browser.");t.startsWith(Fe.URL_SCHEME)&&(t=t.slice(Fe.URL_SCHEME.length)),(t==null||t.length===0)&&(t=bN),this.modelJsonFileName=t+wN,this.weightDataFileName=t+NN}async save(t){if(typeof document>"u")throw new Error("Browser downloads are not supported in this environment since `document` is not present");const n=Ct.join(t.weightData),r=window.URL.createObjectURL(new Blob([n],{type:"application/octet-stream"}));if(t.modelTopology instanceof ArrayBuffer)throw new Error("BrowserDownloads.save() does not support saving model topology in binary formats yet.");{const s=[{paths:["./"+this.weightDataFileName],weights:t.weightSpecs}],a=lc(t,s),o=window.URL.createObjectURL(new Blob([JSON.stringify(a)],{type:"application/json"})),i=this.modelJsonAnchor==null?document.createElement("a"):this.modelJsonAnchor;if(i.download=this.modelJsonFileName,i.href=o,await po(()=>i.dispatchEvent(new MouseEvent("click"))),t.weightData!=null){const u=this.weightDataAnchor==null?document.createElement("a"):this.weightDataAnchor;u.download=this.weightDataFileName,u.href=r,await po(()=>u.dispatchEvent(new MouseEvent("click")))}return{modelArtifactsInfo:An(t)}}}}Fe.URL_SCHEME="downloads://";class vN{constructor(t){if(t==null||t.length<1)throw new Error(`When calling browserFiles, at least 1 file is required, but received ${t}`);this.jsonFile=t[0],this.weightsFiles=t.slice(1)}async load(){return new Promise((t,n)=>{const r=new FileReader;r.onload=s=>{const a=JSON.parse(s.target.result),o=a.modelTopology;if(o==null){n(new Error(`modelTopology field is missing from file ${this.jsonFile.name}`));return}if(a.weightsManifest==null){n(new Error(`weightManifest field is missing from file ${this.jsonFile.name}`));return}if(this.weightsFiles.length===0){t({modelTopology:o});return}const u=Gs(a,l=>this.loadWeights(l));t(u)},r.onerror=s=>n(`Failed to read model topology and weights manifest JSON from file '${this.jsonFile.name}'. BrowserFiles supports loading Keras-style tf.Model artifacts only.`),r.readAsText(this.jsonFile)})}loadWeights(t){const n=[],r=[];for(const o of t)n.push(...o.weights),r.push(...o.paths);const s=this.checkManifestAndWeightFiles(t),a=r.map(o=>this.loadWeightsFile(o,s[o]));return Promise.all(a).then(o=>[n,o])}loadWeightsFile(t,n){return new Promise((r,s)=>{const a=new FileReader;a.onload=o=>{const i=o.target.result;r(i)},a.onerror=o=>s(`Failed to weights data from file of path '${t}'.`),a.readAsArrayBuffer(n)})}checkManifestAndWeightFiles(t){const n=[],r=this.weightsFiles.map(a=>eo(a.name)),s={};for(const a of t)a.paths.forEach(o=>{const i=eo(o);if(n.indexOf(i)!==-1)throw new Error(`Duplicate file basename found in weights manifest: '${i}'`);if(n.push(i),r.indexOf(i)===-1)throw new Error(`Weight file with basename '${i}' is not provided.`);s[o]=this.weightsFiles[r.indexOf(i)]});if(n.length!==this.weightsFiles.length)throw new Error(`Mismatch in the number of files in weights manifest (${n.length}) and the number of weight files provided (${this.weightsFiles.length}).`);return s}}const SN=e=>M().getBool("IS_BROWSER")&&!Array.isArray(e)&&e.startsWith(Fe.URL_SCHEME)?TN(e.slice(Fe.URL_SCHEME.length)):null;rt.registerSaveRouter(SN);function TN(e="model"){return new Fe(e)}function EN(e){return new vN(e)}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fo(e,t,n,r){o(e),n=n??0,r=r??1,i(n,r);let s=0;const a=u=>(u.then(l=>{const h=n+ ++s/e.length*(r-n);return t(h),l}),u);function o(u){y(u!=null&&Array.isArray(u)&&u.length>0,()=>"promises must be a none empty array")}function i(u,l){y(u>=0&&u<=1,()=>`Progress fraction must be in range [0, 1], but got startFraction ${u}`),y(l>=0&&l<=1,()=>`Progress fraction must be in range [0, 1], but got endFraction ${l}`),y(l>=u,()=>`startFraction must be no more than endFraction, but got startFraction ${u} and endFraction ${l}`)}return Promise.all(e.map(a))}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function Hp(e,t){t==null&&(t={});const n=t.fetchFunc==null?M().platform.fetch:t.fetchFunc,r=e.map(c=>n(c,t.requestInit,{isBinary:!0})),i=(t.onProgress==null?await Promise.all(r):await fo(r,t.onProgress,0,.5)).map(c=>c.arrayBuffer());return t.onProgress==null?await Promise.all(i):await fo(i,t.onProgress,.5,1)}function $N(e,t){var n;const r=t.fetchFunc==null?M().platform.fetch:t.fetchFunc;let s=0,a;return(n=t.onProgress)===null||n===void 0||n.call(t,0),new ReadableStream({pull:async o=>{for(var i;s<e.length;){a||(a=(await r(e[s],t.requestInit,{isBinary:!0})).body.getReader());const{done:u,value:l}=await a.read();if(u){s++,a=void 0,(i=t.onProgress)===null||i===void 0||i.call(t,s/e.length);continue}o.enqueue(l);return}o.close()}})}async function _N(e,t="",n,r){return jp(o=>Hp(o,{requestInit:r}))(e,t,n)}function jp(e){return async(t,n="",r)=>{const s=t.map(()=>!1),a={},o=r!=null?r.map(()=>!1):[],i=[];if(t.forEach((d,g)=>{let N=0;d.weights.forEach(w=>{const T="quantization"in w?w.quantization.dtype:w.dtype,x=Ie[T]*K(w.shape),$=()=>{s[g]=!0,a[g]==null&&(a[g]=[]),a[g].push({manifestEntry:w,groupOffset:N,sizeBytes:x})};r!=null?r.forEach((E,I)=>{E===w.name&&($(),o[I]=!0)}):$(),i.push(w.name),N+=x})}),!o.every(d=>d)){const d=r.filter((g,N)=>!o[N]);throw new Error(`Could not find weights in manifest with names: ${d.join(", ")}. 
Manifest JSON has weights with names: ${i.join(", ")}.`)}const u=s.reduce((d,g,N)=>(g&&d.push(N),d),[]),l=[];u.forEach(d=>{t[d].paths.forEach(g=>{const N=n+(n.endsWith("/")?"":"/")+g;l.push(N)})});const h=await e(l),c={};let p=0;return u.forEach(d=>{const g=t[d].paths.length,N=new Ct(h.slice(p,p+g));a[d].forEach(T=>{const x=N.slice(T.groupOffset,T.groupOffset+T.sizeBytes),$=oc(x,[T.manifestEntry]);for(const E in $)c[E]=$[E]}),p+=g}),c}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const kN="application/octet-stream",IN="application/json";class Va{constructor(t,n){if(this.DEFAULT_METHOD="POST",n==null&&(n={}),this.weightPathPrefix=n.weightPathPrefix,this.weightUrlConverter=n.weightUrlConverter,n.fetchFunc!=null?(y(typeof n.fetchFunc=="function",()=>"Must pass a function that matches the signature of `fetch` (see https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)"),this.fetch=n.fetchFunc):this.fetch=M().platform.fetch,y(t!=null&&t.length>0,()=>"URL path for http must not be null, undefined or empty."),Array.isArray(t)&&y(t.length===2,()=>`URL paths for http must have a length of 2, (actual length is ${t.length}).`),this.path=t,n.requestInit!=null&&n.requestInit.body!=null)throw new Error("requestInit is expected to have no pre-existing body, but has one.");this.requestInit=n.requestInit||{},this.loadOptions=n}async save(t){if(t.modelTopology instanceof ArrayBuffer)throw new Error("BrowserHTTPRequest.save() does not support saving model topology in binary formats yet.");const n=Object.assign({method:this.DEFAULT_METHOD},this.requestInit);n.body=new FormData;const r=[{paths:["./model.weights.bin"],weights:t.weightSpecs}],s=lc(t,r);if(n.body.append("model.json",new Blob([JSON.stringify(s)],{type:IN}),"model.json"),t.weightData!=null){const o=Ct.join(t.weightData);n.body.append("model.weights.bin",new Blob([o],{type:kN}),"model.weights.bin")}const a=await this.fetch(this.path,n);if(a.ok)return{modelArtifactsInfo:An(t),responses:[a]};throw new Error(`BrowserHTTPRequest.save() failed due to HTTP response status ${a.status}.`)}async loadModelJSON(){const t=await this.fetch(this.path,this.requestInit);if(!t.ok)throw new Error(`Request to ${this.path} failed with status code ${t.status}. Please verify this URL points to the model JSON of the model to load.`);let n;try{n=await t.json()}catch{let o=`Failed to parse model JSON of response from ${this.path}.`;throw this.path.endsWith(".pb")?o+=" Your path contains a .pb file extension. Support for .pb models have been removed in TensorFlow.js 1.0 in favor of .json models. You can re-convert your Python TensorFlow model using the TensorFlow.js 1.0 conversion scripts or you can convert your.pb models with the 'pb2json'NPM script in the tensorflow/tfjs-converter repository.":o+=" Please make sure the server is serving valid JSON for this request.",new Error(o)}const r=n.modelTopology,s=n.weightsManifest;if(r==null&&s==null)throw new Error(`The JSON from HTTP path ${this.path} contains neither model topology or manifest for weights.`);return n}async load(){if(this.loadOptions.streamWeights)return this.loadStream();const t=await this.loadModelJSON();return Gs(t,n=>this.loadWeights(n))}async loadStream(){const t=await this.loadModelJSON(),n=await this.getWeightUrls(t.weightsManifest),r=cs(t.weightsManifest),s=()=>$N(n,this.loadOptions);return Object.assign(Object.assign({},t),{weightSpecs:r,getWeightStream:s})}async getWeightUrls(t){const n=Array.isArray(this.path)?this.path[1]:this.path,[r,s]=xN(n),a=this.weightPathPrefix||r,o=[],i=[];for(const u of t)for(const l of u.paths)this.weightUrlConverter!=null?i.push(this.weightUrlConverter(l)):o.push(a+l+s);return this.weightUrlConverter&&o.push(...await Promise.all(i)),o}async loadWeights(t){const n=await this.getWeightUrls(t),r=cs(t),s=await Hp(n,this.loadOptions);return[r,s]}}Va.URL_SCHEME_REGEX=/^https?:\/\//;function xN(e){const t=e.lastIndexOf("/"),n=e.lastIndexOf("?"),r=e.substring(0,t),s=n>t?e.substring(n):"";return[r+"/",s]}function ys(e){return e.match(Va.URL_SCHEME_REGEX)!=null}const Gp=(e,t)=>{if(typeof fetch>"u"&&(t==null||t.fetchFunc==null))return null;{let n=!0;if(Array.isArray(e)?n=e.every(r=>ys(r)):n=ys(e),n)return Wa(e,t)}return null};rt.registerSaveRouter(Gp);rt.registerLoadRouter(Gp);function Wa(e,t){return new Va(e,t)}function AN(e,t){return Wa(e,t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Hr{constructor(t){this.modelArtifacts=t}load(){return this.modelArtifacts}}class Kp{constructor(t){this.saveHandler=t}save(t){return this.saveHandler(t)}}class ON{constructor(t){t.load&&(this.load=()=>Promise.resolve(t.load())),t.save&&(this.save=n=>Promise.resolve(t.save(n)))}}function DN(e,t,n,r){const s=arguments;return new ON(Xp(...s))}function Xp(e,t,n,r){return arguments.length===1?e.modelTopology!=null||e.weightSpecs!=null?new Hr(e):(console.warn("Please call tf.io.fromMemory() with only one argument. The argument should be of type ModelArtifacts. The multi-argument signature of tf.io.fromMemory() has been deprecated and will be removed in a future release."),new Hr({modelTopology:e})):(console.warn("Please call tf.io.fromMemory() with only one argument. The argument should be of type ModelArtifacts. The multi-argument signature of tf.io.fromMemory() has been deprecated and will be removed in a future release."),new Hr({modelTopology:e,weightSpecs:t,weightData:n,trainingConfig:r}))}function FN(e){return new Kp(e)}function RN(e){return new Kp(e)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ua=Object.freeze(Object.defineProperty({__proto__:null,CompositeArrayBuffer:Ct,browserFiles:EN,browserHTTPRequest:AN,concatenateArrayBuffers:km,copyModel:Xm,decodeWeights:oc,decodeWeightsStream:uc,encodeWeights:vm,fromMemory:DN,fromMemorySync:Xp,getLoadHandlers:Bm,getModelArtifactsForJSON:Gs,getModelArtifactsForJSONSync:cc,getModelArtifactsInfoForJSON:An,getSaveHandlers:Rm,getWeightSpecs:cs,http:Wa,isHTTPScheme:ys,listModels:Gm,loadWeights:_N,moveModel:Ym,registerLoadRouter:Fm,registerSaveRouter:Dm,removeModel:Km,weightsLoaderFactory:jp,withSaveHandler:FN,withSaveHandlerSync:RN},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function BN(e,t,n){const r=m(e,"labels","confusionMatrix"),s=m(t,"predictions","confusionMatrix");y(n==null||n>0&&Number.isInteger(n),()=>`If provided, numClasses must be a positive integer, but got ${n}`),y(r.rank===1,()=>`Expected the rank of labels to be 1, but got ${r.rank}`),y(s.rank===1,()=>`Expected the rank of predictions to be 1, but got ${s.rank}`),y(r.shape[0]===s.shape[0],()=>`Mismatch in the number of examples: ${r.shape[0]} vs. ${s.shape[0]}. Labels and predictions should have the same number of elements.`),y(n>0&&Number.isInteger(n),()=>`numClasses is required to be a positive integer, but got ${n}`);const a=Tn(j(r,"int32"),n),o=Tn(j(s,"int32"),n),i=$n(a),u=H(i,o);return j(u,"int32")}const PN=v({confusionMatrix_:BN});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const CN=Object.freeze(Object.defineProperty({__proto__:null,confusionMatrix:PN},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let Ne,mo=!1;function Yp(e,t=3){if(t>4)throw new Error("Cannot construct Tensor with more than 4 channels from pixels.");if(e==null)throw new Error("pixels passed to tf.browser.fromPixels() can not be null");let n=!1,r=!1,s=!1,a=!1,o=!1,i=!1;if(e.data instanceof Uint8Array)n=!0;else if(typeof ImageData<"u"&&e instanceof ImageData)r=!0;else if(typeof HTMLVideoElement<"u"&&e instanceof HTMLVideoElement)s=!0;else if(typeof HTMLImageElement<"u"&&e instanceof HTMLImageElement)a=!0;else if(e.getContext!=null)o=!0;else if(typeof ImageBitmap<"u"&&e instanceof ImageBitmap)i=!0;else throw new Error(`pixels passed to tf.browser.fromPixels() must be either an HTMLVideoElement, HTMLImageElement, HTMLCanvasElement, ImageData in browser, or OffscreenCanvas, ImageData in webworker or {data: Uint32Array, width: number, height: number}, but was ${e.constructor.name}`);if(fn(Jr,S.backendName)!=null){const g={pixels:e},N={numChannels:t};return S.runKernel(Jr,g,N)}const[l,h]=s?[e.videoWidth,e.videoHeight]:[e.width,e.height];let c;if(o)c=e.getContext("2d").getImageData(0,0,l,h).data;else if(r||n)c=e.data;else if(a||s||i){if(Ne==null)if(typeof document>"u")if(typeof OffscreenCanvas<"u"&&typeof OffscreenCanvasRenderingContext2D<"u")Ne=new OffscreenCanvas(1,1).getContext("2d");else throw new Error("Cannot parse input in current context. Reason: OffscreenCanvas Context2D rendering is not supported.");else Ne=document.createElement("canvas").getContext("2d",{willReadFrequently:!0});Ne.canvas.width=l,Ne.canvas.height=h,Ne.drawImage(e,0,0,l,h),c=Ne.getImageData(0,0,l,h).data}let p;if(t===4)p=new Int32Array(c);else{const g=l*h;p=new Int32Array(g*t);for(let N=0;N<g;N++)for(let w=0;w<t;++w)p[N*t+w]=c[N*4+w]}return Aa(p,[h,l,t],"int32")}function LN(e){return e!=null&&e.data instanceof Uint8Array}function zN(){return typeof window<"u"&&typeof ImageBitmap<"u"&&window.hasOwnProperty("createImageBitmap")}function MN(e){return e!=null&&e.width!==0&&e.height!==0}function VN(e){return zN()&&!(e instanceof ImageBitmap)&&MN(e)&&!LN(e)}async function WN(e,t=3){let n=null;if(M().getBool("WRAP_TO_IMAGEBITMAP")&&VN(e)){let r;try{r=await createImageBitmap(e,{premultiplyAlpha:"none"})}catch{r=null}r!=null&&r.width===e.width&&r.height===e.height?n=r:n=e}else n=e;return Yp(n,t)}function Zp(e){if(e.rank!==2&&e.rank!==3)throw new Error(`toPixels only supports rank 2 or 3 tensors, got rank ${e.rank}.`);const t=e.rank===2?1:e.shape[2];if(t>4||t===2)throw new Error(`toPixels only supports depth of size 1, 3 or 4 but got ${t}`);if(e.dtype!=="float32"&&e.dtype!=="int32")throw new Error(`Unsupported type for toPixels: ${e.dtype}. Please use float32 or int32 tensors.`)}function UN(e){const t=(e==null?void 0:e.alpha)||1;if(t>1||t<0)throw new Error(`Alpha value ${t} is suppoed to be in range [0 - 1].`)}async function qN(e,t){let n=m(e,"img","toPixels");if(!(e instanceof et)){const l=n;n=j(l,"int32"),l.dispose()}Zp(n);const[r,s]=n.shape.slice(0,2),a=n.rank===2?1:n.shape[2],o=await n.data(),i=n.dtype==="float32"?255:1,u=new Uint8ClampedArray(s*r*4);for(let l=0;l<r*s;++l){const h=[0,0,0,255];for(let p=0;p<a;p++){const d=o[l*a+p];if(n.dtype==="float32"){if(d<0||d>1)throw new Error(`Tensor values for a float32 Tensor must be in the range [0 - 1] but encountered ${d}.`)}else if(n.dtype==="int32"&&(d<0||d>255))throw new Error(`Tensor values for a int32 Tensor must be in the range [0 - 255] but encountered ${d}.`);a===1?(h[0]=d*i,h[1]=d*i,h[2]=d*i):h[p]=d*i}const c=l*4;u[c+0]=Math.round(h[0]),u[c+1]=Math.round(h[1]),u[c+2]=Math.round(h[2]),u[c+3]=Math.round(h[3])}if(t!=null){mo||fn(Cs,S.backendName)!=null&&(console.warn("tf.browser.toPixels is not efficient to draw tensor on canvas. Please try tf.browser.draw instead."),mo=!0),t.width=s,t.height=r;const l=t.getContext("2d"),h=new ImageData(u,s,r);l.putImageData(h,0,0)}return n!==e&&n.dispose(),u}function HN(e,t,n){let r=m(e,"img","draw");if(!(e instanceof et)){const o=r;r=j(o,"int32"),o.dispose()}Zp(r),UN(n==null?void 0:n.imageOptions);const s={image:r},a={canvas:t,options:n};S.runKernel(Cs,s,a)}const Jp=v({fromPixels_:Yp}),jN=Object.freeze(Object.defineProperty({__proto__:null,draw:HN,fromPixels:Jp,fromPixelsAsync:WN,toPixels:qN},Symbol.toStringTag,{value:"Module"}));function Qp(e,t){const n=e.shape.length,r=t.shape.length;if(n<1)throw new Error(`tf.gatherND() expects the input to be rank 1 or higher, but the rank was ${n}.`);if(r<1)throw new Error(`tf.gatherND() expects the indices to be rank 1 or higher, but the rank was ${r}.`);if(t.dtype!=="int32")throw new Error(`tf.gatherND() expects the indices to be int32 type, but the dtype was ${t.dtype}.`);if(t.shape[r-1]>n)throw new Error(`index innermost dimension length must be <= tensor rank; saw: ${t.shape[r-1]} vs. ${n}`);if(K(e.shape)===0)throw new Error(`Requested more than 0 entries, but input is empty. Input shape: ${e.shape}.`);const s=t.shape,a=s[s.length-1];let o=1;for(let c=0;c<s.length-1;++c)o*=s[c];const i=e.shape,u=s.slice();u.pop();let l=1;for(let c=a;c<n;++c)l*=i[c],u.push(i[c]);const h=[...Ze(e.shape).map(c=>c/l),1].slice(0,a);return[u,o,l,h]}const GN=Object.freeze(Object.defineProperty({__proto__:null,prepareAndValidate:Qp},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const bs=-2,KN=-1;function XN(e,t,n){const r=e.shape.length;y(r===t.length,()=>`Error in slice${r}D: Length of begin ${t} must match the rank of the array (${r}).`),y(r===n.length,()=>`Error in slice${r}D: Length of size ${n} must match the rank of the array (${r}).`);for(let s=0;s<r;++s)y(t[s]+n[s]<=e.shape[s],()=>`Error in slice${r}D: begin[${s}] + size[${s}] (${t[s]+n[s]}) would overflow input.shape[${s}] (${e.shape[s]})`)}function YN(e){const t=[];let n=0;for(;e>0;)e&1&&t.push(n),e/=2,n++;return t}function ZN(e,t,n){const r=[];for(let s=0;s<e.length;s++)r[s]=Math.ceil((t[s]-e[s])/n[s]);return r}function tf(e,t,n,r){const s=[...e];for(let a=s.length;a<r.length;a++)s.push(1);for(let a=0;a<n;a++)a===0?s[t]=1:(s.splice(t,0,1),s.pop());return s}function ef(e,t,n){return n<=e?n:n-(t-1)}function nf(e,t){const n=[];for(let r=0;r<e;r++)n.push(t+r);return n}function JN(e,t,n,r,s,a,o,i,u){const l=e.length;let h=new Array(l),c=new Array(l),p=new Array(l);if(t.length&&n>0){const d=t[0],g=n+1;h=rf(o,d,g,r,e),c=sf(i,d,g,s,e),p=tf(a,d,g,e)}else for(let d=0;d<l;d++)h[d]=of(o,r,a,e,d,u),c[d]=uf(i,s,a,e,d,u),p[d]=af(a,d,u);return{begin:h,end:c,strides:p}}function rf(e,t,n,r,s){const a=[...s],o=nf(n,t);for(let i=0;i<a.length;i++)if(o.indexOf(i)>-1)a[i]=0;else{const u=ef(t,n,i);let l=r[u];e&1<<u&&(l=0),a[i]=l}return a}function sf(e,t,n,r,s){const a=[...s],o=nf(n,t);for(let i=0;i<a.length;i++)if(o.indexOf(i)>-1)a[i]=Number.MAX_SAFE_INTEGER;else{const u=ef(t,n,i);let l=r[u];e&1<<u&&(l=Number.MAX_SAFE_INTEGER),a[i]=l}for(let i=0;i<a.length;i++){const u=s[i];a[i]<0&&(a[i]+=u),a[i]=hn(0,a[i],s[i])}return a}function af(e,t,n){let r=e[t];return(n&1<<t||r==null)&&(r=1),r}function of(e,t,n,r,s,a){let o=t[s];const i=n[s]||1;(e&1<<s||a&1<<s||o==null)&&(i>0?o=Number.MIN_SAFE_INTEGER:o=Number.MAX_SAFE_INTEGER);const u=r[s];return o<0&&(o+=u),o=hn(0,o,u-1),o}function uf(e,t,n,r,s,a){let o=t[s];const i=n[s]||1;(e&1<<s||a&1<<s||o==null)&&(i>0?o=Number.MAX_SAFE_INTEGER:o=Number.MIN_SAFE_INTEGER);const u=r[s];return o<0&&(o+=u),i>0?o=hn(0,o,u):o=hn(-1,o,u-1),o}function QN(e,t,n){let r=n.length;for(let s=0;s<n.length;s++)if(n[s]>1){r=s;break}for(let s=r+1;s<n.length;s++)if(t[s]>0||n[s]!==e[s])return!1;return!0}function tv(e,t){let n=e.length>0?e[e.length-1]:1;for(let r=0;r<e.length-1;r++)n+=e[r]*t[r];return n}function ev(e,t,n){let r;const s=e.shape.length;typeof t=="number"?r=[t,...new Array(s-1).fill(0)]:t.length<s?r=t.concat(new Array(s-t.length).fill(0)):r=t.slice(),r.forEach(o=>{y(o!==-1,()=>"slice() does not support negative begin indexing.")});let a;return n==null?a=new Array(s).fill(-1):typeof n=="number"?a=[n,...new Array(s-1).fill(-1)]:n.length<s?a=n.concat(new Array(s-n.length).fill(-1)):a=n,a=a.map((o,i)=>o>=0?o:(y(o===-1,()=>`Negative size values should be exactly -1 but got ${o} for the slice() size at index ${i}.`),e.shape[i]-r[i])),[r,a]}function nv(e,t,n,r,s,a,o,i,u){let l;if(r==null?(l=new Array(t.length),l.fill(1)):l=r,o!=null&&(o&o-1)!==0)throw new Error("Multiple ellipses in slice is not allowed.");let h=!1;const c={dims:l.length,numAddAxisAfterEllipsis:0,begin:t.slice(),end:n.slice(),strides:l.slice(),beginMask:s,endMask:a,ellipsisMask:o,newAxisMask:i,shrinkAxisMask:u};for(let $=0;$<c.dims;$++)h&&(1<<$&i)!==0&&c.numAddAxisAfterEllipsis++,1<<$&o&&(h=!0);h||(c.ellipsisMask|=1<<c.dims,c.dims++);const p={dims:e.length,beginMask:0,endMask:0,beginValid:!1,endValid:!1};rv(c,p);let d=!0,g=!0,N=!0;const w=[],T=[];for(let $=0;$<e.length;++$){if(p.strides[$]===0)throw Error(`strides[${$}] must be non-zero`);const E=!!(p.shrinkAxisMask&1<<$),I=e[$];if(I===-1){w.push(E?1:-1);continue}const A=[p.beginMask&1<<$,p.endMask&1<<$],F=[p.strides[$]>0?0:-1,p.strides[$]>0?I:I-1];if(E&&p.strides[$]<=0)throw Error("only stride 1 allowed on non-range indexing.");N=N&&p.strides[$]===1;const R=!!(p.beginMask&1<<$&&p.endMask&1<<$);if(p.beginValid&&p.endValid){if(E){const D=p.begin[$]<0?I+p.begin[$]:p.begin[$];if(p.begin[$]=D,p.end[$]=p.begin[$]+1,D<0||D>=I)throw Error(`slice index ${p.begin[$]} of dimension ${$} out of bounds.`)}else p.begin[$]=go(p.begin[$],0,p.strides[$],I,A,F),p.end[$]=go(p.end[$],1,p.strides[$],I,A,F);const b=p.strides[$]===1&&p.begin[$]===0&&p.end[$]===I;d=d&&b,g=g&&($===0&&p.strides[$]===1||b)}else d=d&&p.strides[$]===1&&R,g=g&&($===0&&p.strides[$]===1||R);let k,_=!1;if(p.beginValid&&p.endValid?(k=p.end[$]-p.begin[$],_=!0):E?(k=1,_=!0):R&&I>=0&&(p.strides[$]<0?k=-I:k=I,_=!0),_){let b;k===0||k<0!=p.strides[$]<0?b=0:b=Math.trunc(k/p.strides[$])+(k%p.strides[$]!==0?1:0),w.push(b)}else w.push(-1)}for(let $=0;$<p.finalShapeGatherIndices.length;++$){const E=p.finalShapeGatherIndices[$];E>=0?T.push(w[E]):E===bs&&T.push(1)}return{finalShapeSparse:T.filter(($,E)=>p.finalShapeGatherIndices[E]!==bs),finalShape:T,isIdentity:d,sliceDim0:g,isSimpleSlice:N,begin:p.begin,end:p.end,strides:p.strides}}function rv(e,t){t.beginMask=0,t.endMask=0,t.shrinkAxisMask=0;let n=0;t.beginValid=e.begin!=null,t.endValid=e.end!=null,t.begin=new Array(t.dims),t.end=new Array(t.dims),t.strides=new Array(t.dims),t.finalShapeGatherIndices=[],t.finalShapeGatherIndicesSparse=[],t.inputShapeGatherIndicesSparse=new Array(t.dims);for(let r=0;r<e.dims;r++)if(1<<r&e.ellipsisMask){const s=Math.min(t.dims-(e.dims-r)+1+e.numAddAxisAfterEllipsis,t.dims);for(;n<s;n++)t.begin[n]=0,t.end[n]=0,t.strides[n]=1,t.beginMask|=1<<n,t.endMask|=1<<n,t.finalShapeGatherIndices.push(n),t.finalShapeGatherIndicesSparse.push(-1),t.inputShapeGatherIndicesSparse[n]=r}else if(1<<r&e.newAxisMask)t.finalShapeGatherIndices.push(bs),t.finalShapeGatherIndicesSparse.push(-1);else{if(n===t.begin.length)throw Error(`Index out of range using input dim ${n}; input has only ${t.dims} dims, ${t.begin.length}.`);e.begin!=null&&(t.begin[n]=e.begin[r]),e.end!=null&&(t.end[n]=e.end[r]),t.strides[n]=e.strides[r],e.beginMask&1<<r&&(t.beginMask|=1<<n),e.endMask&1<<r&&(t.endMask|=1<<n),e.shrinkAxisMask&1<<r?(t.finalShapeGatherIndices.push(KN),t.finalShapeGatherIndicesSparse.push(-1),t.shrinkAxisMask|=1<<n):(t.finalShapeGatherIndices.push(n),t.finalShapeGatherIndicesSparse.push(r)),t.inputShapeGatherIndicesSparse[n]=r,n++}}function go(e,t,n,r,s,a){if(s[t])return n>0?a[t]:a[t+1&1];{const o=e<0?r+e:e;return o<a[0]?a[0]:o>a[1]?a[1]:o}}const lf=Object.freeze(Object.defineProperty({__proto__:null,assertParamsValid:XN,computeFlatOffset:tv,computeOutShape:ZN,getNormalizedAxes:JN,isSliceContinous:QN,maskToAxes:YN,parseSliceParams:ev,sliceInfo:nv,startForAxis:of,startIndicesWithElidedDims:rf,stopForAxis:uf,stopIndicesWithElidedDims:sf,stridesForAxis:af,stridesWithElidedDims:tf},Symbol.toStringTag,{value:"Module"}));/** @license See the LICENSE file. */const sv="4.22.0";/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class cf{static sgd(t){return new Cr(t)}static momentum(t,n,r=!1){return new za(t,n,r)}static rmsprop(t,n=.9,r=0,s=null,a=!1){return new Ma(t,n,r,s,a)}static adam(t=.001,n=.9,r=.999,s=null){return new Ca(t,n,r,s)}static adadelta(t=.001,n=.95,r=null){return new Ba(t,n,r)}static adamax(t=.002,n=.9,r=.999,s=null,a=0){return new La(t,n,r,s,a)}static adagrad(t,n=.1){return new Pa(t,n)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const av=cf;/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ov=typeof requestAnimationFrame<"u"?requestAnimationFrame:typeof setImmediate<"u"?setImmediate:e=>e();function iv(){return new Promise(e=>ov(()=>e()))}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function uv(e,t){const n=e[0].length;e.forEach((s,a)=>{y(s.length===n,()=>`Error in concat${n}D: rank of tensors[${a}] must be the same as the rank of the rest (${n})`)}),y(t>=0&&t<n,()=>`Error in concat${n}D: axis must be between 0 and ${n-1}.`);const r=e[0];e.forEach((s,a)=>{for(let o=0;o<n;o++)y(o===t||s[o]===r[o],()=>`Error in concat${n}D: Shape of tensors[${a}] (${s}) does not match the shape of the rest (${r}) along the non-concatenated axis ${a}.`)})}function lv(e,t){const n=e[0].slice();for(let r=1;r<e.length;r++)n[t]+=e[r][t];return n}/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var zt;(function(e){e[e.FIRST_DIM_SIZE=0]="FIRST_DIM_SIZE",e[e.VALUE_ROWIDS=1]="VALUE_ROWIDS",e[e.ROW_LENGTHS=2]="ROW_LENGTHS",e[e.ROW_SPLITS=3]="ROW_SPLITS",e[e.ROW_LIMITS=4]="ROW_LIMITS",e[e.ROW_STARTS=5]="ROW_STARTS"})(zt||(zt={}));function cv(e,t,n){let r=new Array;if(n==null&&t==null)return r;if(t==null)for(;r.length<e+n.length;)r.push(-1);else r=t.slice();if(n==null)return r;if(e+n.length!==r.length)throw new Error(`rt input.shape and shape=${t} are incompatible: rt input.rank = ${e+n.length}, but shape.rank = ${r.length}`);for(let s=1;s<n.length;++s){const a=n[s],o=r[r.length-n.length+s],i=r[o];if(a>=0)if(i>=0){if(i!==a)throw new Error(`rt input.shape and shape=${t} are incompatible: rt input.shape[${s+e}] = ${a} but shape[${s+e}] = ${i}`)}else r[o]=a}return r}function hv(e){const t={FIRST_DIM_SIZE:zt.FIRST_DIM_SIZE,VALUE_ROWIDS:zt.VALUE_ROWIDS,ROW_LENGTHS:zt.ROW_LENGTHS,ROW_SPLITS:zt.ROW_SPLITS,ROW_LIMITS:zt.ROW_LIMITS,ROW_STARTS:zt.ROW_STARTS},n=[];for(const r of e)if(r in t)n.push(t[r]);else break;return n}function pv(e){return e.length===0?0:e[0]===zt.FIRST_DIM_SIZE?e.length-1:e.length}function fv(e,t){if(e==null||t==null)return;const n=e.length,r=t.length;if(n>=r)throw new Error(`defaultValue.shape=${e} and ragged tensor flatValues.shape=${t}, are incompatible: defaultValue.rank = ${n} must be less than ragged tensor input flatValues.rank = ${r})`);for(let s=0;s<Math.min(n,r-1);++s){const a=e[s],o=t[s+1];if(a>=0&&o>=0&&a!==1&&a!==o)throw new Error(`defaultValue.shape=${e}, and ragged tensor input flatValues.shape=${t} are incompatible: defaultValue.shape[${s-e.length}] = ${a} but ragged tensor input.flatValues.shape[${s-e.length}] = ${o}`)}}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const qa=30;function dv(e){return e<=qa?e:hr(e,Math.floor(Math.sqrt(e)))}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function mv(e,t,n){const r=n*(typeof e=="number"?e:e[0]),s=t*(typeof e=="number"?e:e[1]);return[r,s]}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function gv(e,t,n,r=!0){let s=[];if(r)s=s.concat(t.slice(0)),s.push(e[0]/n),s=s.concat(e.slice(1));else{s=s.concat(e[0]);const a=t.length;for(let o=0;o<a;++o)s=s.concat([e[o+1]/t[o],t[o]]);s=s.concat(e.slice(a+1))}return s}function yv(e,t,n=!0){const r=[];if(n){r.push(t);for(let s=t+1;s<e;++s)s<=2*t?(r.push(s),r.push(s-(t+1))):r.push(s)}else{const s=[],a=[];for(let o=1;o<e;++o)o>=t*2+1||o%2===1?a.push(o):s.push(o);r.push(...s),r.push(0),r.push(...a)}return r}function bv(e,t,n,r=!0){const s=[];r?s.push(e[0]/n):s.push(e[0]*n);for(let a=1;a<e.length;++a)a<=t.length?r?s.push(t[a-1]*e[a]):s.push(e[a]/t[a-1]):s.push(e[a]);return s}function wv(e,t){const n=[0];for(let r=0;r<t;++r)n.push(e[r][0]);return n}function Nv(e,t,n){const r=e.slice(0,1);for(let s=0;s<n;++s)r.push(e[s+1]-t[s][0]-t[s][1]);return r}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const vv=1.7580993408473768,Sv=1.0507009873554805;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Tv=.3275911,Ev=.254829592,$v=-.284496736,_v=1.421413741,kv=-1.453152027,Iv=1.061405429;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xv(e,t){if(e.length!==t.length)throw new Error(`Cannot merge real and imag arrays of different lengths. real:${e.length}, imag: ${t.length}.`);const n=new Float32Array(e.length*2);for(let r=0;r<n.length;r+=2)n[r]=e[r/2],n[r+1]=t[r/2];return n}function Av(e){const t=new Float32Array(e.length/2),n=new Float32Array(e.length/2);for(let r=0;r<e.length;r+=2)t[r/2]=e[r],n[r/2]=e[r+1];return{real:t,imag:n}}function Ov(e){const t=Math.ceil(e.length/4),n=new Float32Array(t),r=new Float32Array(t);for(let s=0;s<e.length;s+=4)n[Math.floor(s/4)]=e[s],r[Math.floor(s/4)]=e[s+1];return{real:n,imag:r}}function Dv(e){const t=Math.floor(e.length/4),n=new Float32Array(t),r=new Float32Array(t);for(let s=2;s<e.length;s+=4)n[Math.floor(s/4)]=e[s],r[Math.floor(s/4)]=e[s+1];return{real:n,imag:r}}function Fv(e,t){const n=e[t*2],r=e[t*2+1];return{real:n,imag:r}}function Rv(e,t,n,r){e[r*2]=t,e[r*2+1]=n}function Bv(e,t){const n=new Float32Array(e/2),r=new Float32Array(e/2);for(let s=0;s<Math.ceil(e/2);s++){const a=(t?2:-2)*Math.PI*(s/e);n[s]=Math.cos(a),r[s]=Math.sin(a)}return{real:n,imag:r}}function Pv(e,t,n){const r=(n?2:-2)*Math.PI*(e/t),s=Math.cos(r),a=Math.sin(r);return{real:s,imag:a}}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const jr="->",Cv=/->/g,yo=",",bo="...";function Lv(e,t){e=e.replace(/\s/g,"");const n=(e.length-e.replace(Cv,"").length)/jr.length;if(n<1)throw new Error("Equations without an arrow are not supported.");if(n>1)throw new Error(`Equation must contain exactly one arrow ("${jr}").`);const[r,s]=e.split(jr);y(r.indexOf(bo)===-1,()=>`The ellipsis notation ("${bo}") is not supported yet.`);const a=r.split(yo),o=a.length;if(t!==o)throw new Error(`Expected ${o} input tensors, received ${t}`);if(o>2)throw new Error("Support for more than 2 input tensors is not implemented yet.");const i=[];for(let p=0;p<s.length;++p){const d=s[p];if(!a.some(g=>g.indexOf(d)!==-1))throw new Error(`Output subscripts contain the label ${d} not present in the input subscripts.`);i.indexOf(d)===-1&&i.push(d)}for(let p=0;p<r.length;++p){const d=r[p];i.indexOf(d)===-1&&d!==yo&&i.push(d)}const u=new Array(a.length);for(let p=0;p<o;++p){if(new Set(a[p].split("")).size!==a[p].length)throw new Error(`Found duplicate axes in input component ${a[p]}. Support for duplicate axes in input is not implemented yet.`);u[p]=[];for(let d=0;d<a[p].length;++d)u[p].push(i.indexOf(a[p][d]))}const l=i.length,h=s.length,c=[];for(let p=h;p<l;++p)c.push(p);return{allDims:i,summedDims:c,idDims:u}}function zv(e,t){let n=new Array(e);n.fill(-1);for(let s=0;s<t.length;++s)n[t[s]]=s;const r=[];for(let s=0;s<e;++s)n[s]===-1&&r.push(s);return n=n.filter(s=>s!==-1),{permutationIndices:n,expandDims:r}}function Mv(e,t,n){const r=new Array(e);for(let s=0;s<n.length;++s){const a=n[s].shape;for(let o=0;o<t[s].length;++o)r[t[s][o]]===void 0?r[t[s][o]]=a[o]:y(r[t[s][o]]===a[o],()=>`Expected dimension ${r[t[s][o]]} at axis ${o} of input shaped ${JSON.stringify(a)}, but got dimension ${a[o]}`)}}function Vv(e,t){const n=e,r=[];let s=0;e.length===0&&n.push(-1),s=e.length+1;for(let o=0;o<s;++o)r.push([]);const a=[];for(let o=0;o<n.length;++o){const i=n[o],u=Uv(t,i);for(const l of u)a.indexOf(l)===-1&&(r[o].push(l),a.push(l))}return{path:n,steps:r}}function Wv(e){return e.every((t,n)=>t===n)}function Uv(e,t){const n=[];for(let r=0;r<e.length;++r)(e[r].length===0||e[r].indexOf(t)!==-1||t===-1)&&n.push(r);return n}function qv(e,t,n=0){let r=[];if(typeof t=="number")y(e.shape[n]%t===0,()=>"Number of splits must evenly divide the axis."),r=new Array(t).fill(e.shape[n]/t);else{const s=t.reduce((o,i)=>(i===-1&&(o+=1),o),0);y(s<=1,()=>"There should be only one negative value in split array.");const a=t.indexOf(-1);if(a!==-1){const o=t.reduce((i,u)=>u>0?i+u:i);t[a]=e.shape[n]-o}y(e.shape[n]===t.reduce((o,i)=>o+i),()=>"The sum of sizes must match the size of the axis dimension."),r=t}return r}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hv(e){return`Received SparseTensor with denseShape[0] = 0 but
  indices.shape[0] = ${e}`}function jv(e,t){return`indices(${e}, 0) is invalid: ${t} < 0`}function Gv(e,t,n){return`indices(${e}, 0) is invalid: ${t} >= ${n}`}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Kv(e,t){return`only one output dimension may be -1, not both ${e} and ${t}`}function Xv(e,t){return`size ${e} must be non-negative, not ${t}`}function Yv(){return"reshape cannot infer the missing input size for an empty tensor unless all specified input sizes are non-zero"}function Zv(e,t){const n=K(e),r=K(t);return`Input to reshape is a SparseTensor with ${n}
  dense values, but the requested shape requires a multiple of ${r}. inputShape=${e} outputShape= ${t}`}function Jv(e,t){const n=K(e),r=K(t);return`Input to reshape is a tensor with ${n} dense values, but the requested shape has ${r}. inputShape=${e} outputShape=${t}`}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qv(){return"segment ids must be >= 0"}function tS(){return"segment ids are not increasing"}function eS(e,t){return`Segment id ${e} out of range [0, ${t}), possibly because segmentIds input is not sorted.`}function nS(e,t,n){return`Bad: indices[${e}] == ${t} out of range [0, ${n})`}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function rS(e,t){let n=!1,r;for(e<=qa?(r=e,n=!0):r=hr(e,Math.floor(Math.sqrt(e)));!n;)r>t||r===e?n=!0:r=hr(e,r+1);return r}function sS(e,t,n){const r=[],s=e.length;for(let a=0;a<s;a++)a!==t?r.push(e[a]):r.push(n);return r}function aS(e,t,n,r){const s=t.shape.length,a=e.shape.length;if(r!==0&&(r<-s||r>s))throw new Error(`Expect batchDims in the range of [-${s}, ${s}], but got ${r}`);if(r<0&&(r+=s),r>a)throw new Error(`batchDims (${r}) must be less than rank(x) (
    ${a}).`);if(n<r)throw new Error(`batchDims (${r}) must be less than or equal to axis (${n}).`);for(let c=0;c<r;++c)if(e.shape[c]!==t.shape[c])throw new Error(`x.shape[${c}]: ${e.shape[c]} should be equal to indices.shape[${c}]: ${t.shape[c]}.`);const o=e.shape[n],i=[];let u=1,l=1,h=1;for(let c=0;c<r;++c)i.push(e.shape[c]),u*=e.shape[c];for(let c=r;c<n;c++)i.push(e.shape[c]),l*=e.shape[c];for(let c=r;c<s;c++)i.push(t.shape[c]);for(let c=n+1;c<a;c++)i.push(e.shape[c]),h*=e.shape[c];return{batchSize:u,sliceSize:h,outerSize:l,dimSize:o,outputShape:i}}const oS=Object.freeze(Object.defineProperty({__proto__:null,collectGatherOpShapeInfo:aS,computeOutShape:sS,segOpComputeOptimalWindowSize:rS},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function iS(e){try{return e.map(t=>fr(t))}catch(t){throw new Error(`Failed to decode encoded string bytes into utf-8, error: ${t}`)}}function uS(e){return e.map(t=>xn(t))}const lS=Object.freeze(Object.defineProperty({__proto__:null,ERF_A1:Ev,ERF_A2:$v,ERF_A3:_v,ERF_A4:kv,ERF_A5:Iv,ERF_P:Tv,PARALLELIZE_THRESHOLD:qa,get RowPartitionType(){return zt},SELU_SCALE:Sv,SELU_SCALEALPHA:vv,applyActivation:Br,assertAndGetBroadcastShape:at,assertAxesAreInnerMostDims:$y,assertParamsConsistent:uv,assignToTypedArray:Rv,axesAreInnerMostDims:ra,calculateShapes:mp,checkEinsumDimSizes:Mv,checkPadOnDimRoundingMode:Ot,combineLocations:lh,combineRaggedTensorToTensorShapes:cv,complexWithEvenIndex:Ov,complexWithOddIndex:Dv,computeConv2DInfo:On,computeConv3DInfo:Ac,computeDefaultPad:Zs,computeDilation2DInfo:Ng,computeOptimalWindowSize:dv,computeOutAndReduceShapes:Ey,computeOutShape:lv,computePool2DInfo:xc,computePool3DInfo:vg,convertConv2DDataFormat:Oc,decodeEinsumEquation:Lv,eitherStridesOrDilationsAreOne:ne,expandShapeToKeepDim:Bn,exponent:Pv,exponents:Bv,fromStringArrayToUint8:uS,fromUint8ToStringArray:iS,getAxesPermutation:_y,getBroadcastDims:sh,getComplexWithIndex:Fv,getEinsumComputePath:Vv,getEinsumPermutation:zv,getFusedBiasGradient:Rr,getFusedDyActivation:Fr,getImageCenter:mv,getInnerMostAxes:Iy,getPermuted:yv,getRaggedRank:pv,getReductionAxes:ea,getReshaped:gv,getReshapedPermuted:bv,getRowPartitionTypesHelper:hv,getSliceBeginCoords:wv,getSliceSize:Nv,getSparseFillEmptyRowsIndicesDenseShapeMismatch:Hv,getSparseFillEmptyRowsNegativeIndexErrorMessage:jv,getSparseFillEmptyRowsOutOfRangeIndexErrorMessage:Gv,getSparseReshapeEmptyTensorZeroOutputDimErrorMessage:Yv,getSparseReshapeInputOutputMismatchErrorMessage:Jv,getSparseReshapeInputOutputMultipleErrorMessage:Zv,getSparseReshapeMultipleNegativeOneOutputDimErrorMessage:Kv,getSparseReshapeNegativeOutputDimErrorMessage:Xv,getSparseSegmentReductionIndicesOutOfRangeErrorMessage:nS,getSparseSegmentReductionNegativeSegmentIdsErrorMessage:Qv,getSparseSegmentReductionNonIncreasingSegmentIdsErrorMessage:tS,getSparseSegmentReductionSegmentIdOutOfRangeErrorMessage:eS,getUndoAxesPermutation:ky,isIdentityPermutation:Wv,log:Nd,mergeRealAndImagArrays:xv,prepareAndValidate:Qp,prepareSplitSize:qv,segment_util:oS,shouldFuse:Pr,slice_util:lf,splitRealAndImagArrays:Av,stridesOrDilationsArePositive:Oe,tupleValuesAreOne:wn,upcastType:Tr,validateDefaultValueShape:fv,validateInput:Or,validateUpdateShape:Oa,warn:se},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const cS=Object.freeze(Object.defineProperty({__proto__:null,nonMaxSuppressionV3Impl:Rp,nonMaxSuppressionV4Impl:Bp,nonMaxSuppressionV5Impl:Pp,whereImpl:Tp},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */yN();const hf=Object.freeze(Object.defineProperty({__proto__:null,Abs:ni,Acos:ri,Acosh:si,AdadeltaOptimizer:Ba,AdagradOptimizer:Pa,AdamOptimizer:Ca,AdamaxOptimizer:La,Add:Bs,AddN:ai,All:oi,Any:ii,ArgMax:ui,ArgMin:li,Asin:ci,Asinh:hi,Atan:pi,Atan2:di,Atanh:fi,AvgPool:mi,AvgPool3D:gi,AvgPool3DGrad:nd,AvgPoolGrad:ed,BatchMatMul:yi,BatchToSpaceND:bi,Bincount:wi,BitwiseAnd:Ni,BroadcastArgs:vi,BroadcastTo:rd,Cast:Ps,Ceil:Si,ClipByValue:Ti,Complex:Ei,ComplexAbs:$i,Concat:_i,Conv2D:ki,Conv2DBackpropFilter:Ii,Conv2DBackpropInput:xi,Conv3D:Ai,Conv3DBackpropFilterV2:sd,Conv3DBackpropInputV2:Oi,Cos:Di,Cosh:Fi,CropAndResize:Pi,Cumprod:Ri,Cumsum:Bi,DataStorage:Ff,DenseBincount:Ci,DepthToSpace:Li,DepthwiseConv2dNative:zi,DepthwiseConv2dNativeBackpropFilter:Mi,DepthwiseConv2dNativeBackpropInput:Vi,Diag:Wi,Dilation2D:Ui,Dilation2DBackpropFilter:od,Dilation2DBackpropInput:ad,Draw:Cs,get ENV(){return Fs},Einsum:Hi,Elu:ji,EluGrad:id,Environment:ti,Equal:Ki,Erf:Gi,Exp:Xi,ExpandDims:Yi,Expm1:Zi,FFT:Ji,Fill:Qi,FlipLeftRight:tu,Floor:eu,FloorDiv:nu,FromPixels:Jr,FusedBatchNorm:ru,FusedConv2D:ts,FusedDepthwiseConv2D:es,GatherNd:au,GatherV2:su,Greater:ou,GreaterEqual:iu,IFFT:uu,Identity:Ls,Imag:lu,IsFinite:cu,IsInf:hu,IsNan:pu,KernelBackend:Uo,LRN:Su,LRNGrad:hd,LeakyRelu:fu,Less:du,LessEqual:mu,LinSpace:gu,Log:yu,Log1p:bu,LogSoftmax:ld,LogicalAnd:wu,LogicalNot:Nu,LogicalOr:vu,LogicalXor:ud,LowerBound:cd,MatrixBandPart:pd,Max:Tu,MaxPool:$u,MaxPool3D:_u,MaxPool3DGrad:dd,MaxPoolGrad:fd,MaxPoolWithArgmax:ku,Maximum:Eu,Mean:Iu,Min:xu,Minimum:Au,MirrorPad:Ou,Mod:Du,MomentumOptimizer:za,Multinomial:Fu,Multiply:Ru,Neg:Bu,NonMaxSuppressionV3:Cu,NonMaxSuppressionV4:Lu,NonMaxSuppressionV5:zu,NotEqual:Pu,OP_SCOPE_SUFFIX:Ws,OneHot:Vu,OnesLike:Mu,Optimizer:we,OptimizerConstructors:cf,Pack:Wu,PadV2:Uu,Pool:md,Pow:qu,Prelu:Hu,Prod:ju,RMSPropOptimizer:Ma,RaggedGather:Gu,RaggedRange:Ku,RaggedTensorToTensor:Xu,Range:Yu,get Rank(){return ss},Real:Zu,RealDiv:qi,Reciprocal:Ju,get Reduction(){return dt},Relu:Qu,Relu6:rl,Reshape:tl,ResizeBilinear:nl,ResizeBilinearGrad:yd,ResizeNearestNeighbor:el,ResizeNearestNeighborGrad:gd,Reverse:sl,RotateWithOffset:ql,Round:al,Rsqrt:ol,SGDOptimizer:Cr,ScatterNd:il,SearchSorted:ll,Select:cl,Selu:hl,Sigmoid:gl,Sign:ml,Sin:fl,Sinh:dl,Slice:pl,Softmax:Sl,Softplus:yl,SpaceToBatchND:Nl,SparseFillEmptyRows:Tl,SparseReshape:El,SparseSegmentMean:$l,SparseSegmentSum:_l,SparseToDense:kl,SplitV:vl,Sqrt:bl,Square:bd,SquaredDifference:Il,StaticRegexReplace:xl,Step:Ul,StridedSlice:Al,StringNGrams:Ol,StringSplit:Dl,StringToHashBucketFast:Fl,Sub:Rl,Sum:wl,Tan:Bl,Tanh:Pl,Tensor:et,TensorBuffer:dr,TensorScatterUpdate:ul,Tile:zs,TopK:Cl,Transform:Ll,Transpose:Jn,Unique:zl,Unpack:Ml,UnsortedSegmentSum:Vl,UpperBound:wd,Variable:mn,ZerosLike:Wl,_FusedMatMul:Qr,abs:Tt,acos:bc,acosh:wc,add:z,addN:Nc,all:vc,any:Sc,argMax:Ys,argMin:Tc,asin:Ec,asinh:$c,atan:_c,atan2:kc,atanh:Ic,avgPool:Js,avgPool3d:Dc,backend:Hs,backend_util:lS,basicLSTMCell:Fc,batchNorm:Dn,batchNorm2d:Rc,batchNorm3d:Bc,batchNorm4d:Pc,batchToSpaceND:Qs,bincount:ta,bitwiseAnd:Cc,booleanMaskAsync:Ep,broadcastArgs:Lc,broadcastTo:cn,broadcast_util:dy,browser:jN,buffer:qt,cast:j,ceil:zc,clipByValue:Mc,clone:Jt,complex:ee,concat:ht,concat1d:Vc,concat2d:Wc,concat3d:Uc,concat4d:qc,conv1d:Hc,conv2d:Fn,conv2dTranspose:Gc,conv3d:Kc,conv3dTranspose:Xc,copyRegisteredKernels:Ed,cos:Yc,cosh:Zc,cosineWindow:Dr,cumprod:Jc,cumsum:Qc,customGrad:jt,denseBincount:th,deprecationWarn:lm,depthToSpace:eh,depthwiseConv2d:Er,device_util:rm,diag:nh,dilation2d:rh,disableDeprecationWarnings:um,dispose:mt,disposeVariables:cm,div:Y,divNoNan:ah,dot:oh,dropout:xp,einsum:Te,elu:na,enableDebugMode:im,enableProdMode:om,enclosingPowerOfTwo:Fa,engine:Us,ensureShape:ih,env:M,equal:Rn,erf:uh,euclideanNorm:hh,exp:de,expandDims:_t,expm1:ph,eye:sa,fft:xr,fill:Je,findBackend:ym,findBackendFactory:bm,floor:aa,floorDiv:Xs,fused:Op,gather:oa,gatherND:Ip,gather_util:GN,getBackend:qs,getGradient:ns,getKernel:fn,getKernelsForBackend:pr,grad:nb,grads:rb,greater:Qe,greaterEqual:ia,ifft:En,imag:Cn,image:zn,inTopKAsync:Ap,io:Ua,irfft:ka,isFinite:fh,isInf:dh,isNaN:mh,keep:Rt,kernel_impls:cS,leakyRelu:ua,less:yr,lessEqual:$r,linalg:zp,linspace:gh,localResponseNormalization:yh,log:Ke,log1p:la,logSigmoid:wh,logSoftmax:Nh,logSumExp:ha,logicalAnd:Nn,logicalNot:pa,logicalOr:fa,logicalXor:vh,losses:Mp,lowerBound:Sh,matMul:H,math:CN,max:ke,maxPool:da,maxPool3d:Th,maxPoolWithArgmax:Eh,maximum:ma,mean:vn,memory:hm,meshgrid:$h,min:gr,minimum:Sn,mirrorPad:_h,mod:kh,moments:Ih,movingAverage:$p,mul:B,multiRNNCell:xh,multinomial:Ah,neg:Bt,nextFrame:iv,norm:Pn,notEqual:ga,oneHot:Tn,ones:ue,onesLike:Oh,op:v,outerProduct:Dh,pad:tn,pad1d:Fh,pad2d:Rh,pad3d:ya,pad4d:Bh,pool:Ph,pow:Ge,prelu:wa,print:Ks,prod:Ch,profile:pm,raggedGather:Lh,raggedRange:zh,raggedTensorToTensor:Mh,rand:Vh,randomGamma:qh,randomNormal:Ea,randomStandardNormal:Hh,randomUniform:Ir,randomUniformInt:jh,range:me,ready:mm,real:Xe,reciprocal:Gh,registerBackend:wm,registerGradient:vd,registerKernel:Hl,relu:Ln,relu6:$a,removeBackend:gm,reshape:O,reverse:ge,reverse1d:Kh,reverse2d:Xh,reverse3d:Yh,reverse4d:Zh,rfft:Ar,round:_a,rsqrt:Jh,scalar:U,scatterND:_p,scatter_util:iw,searchSorted:kr,selu:Qh,separableConv2d:tp,serialization:mN,setBackend:dm,setPlatform:Nm,setdiff1dAsync:ep,sigmoid:Qt,sign:np,signal:Lp,sin:rp,sinh:sp,slice:X,slice1d:ap,slice2d:op,slice3d:ip,slice4d:up,slice_util:lf,softmax:lp,softplus:ca,spaceToBatchND:ba,sparse:Vp,sparseToDense:kp,spectral:Cp,split:Ye,sqrt:Ht,square:At,squaredDifference:Ia,squeeze:Mt,stack:Gt,step:xa,stridedSlice:cp,string:Wp,sub:W,sum:Q,sumOutType:Kd,tan:hp,tanh:mr,tensor:xt,tensor1d:kt,tensor2d:Ue,tensor3d:Aa,tensor4d:pp,tensor5d:fp,tensor6d:dp,tensorScatterUpdate:gp,tensor_util:Zd,test_util:N0,tidy:V,tile:We,time:fm,topk:yp,train:av,transpose:$n,truncatedNormal:bp,unique:wp,unregisterGradient:Td,unregisterKernel:Sd,unsortedSegmentSum:Np,unstack:be,upcastType:Tr,upperBound:vp,util:Cd,valueAndGrad:sb,valueAndGrads:ab,variable:Sp,variableGrads:bh,version_core:sv,where:te,whereAsync:Da,zeros:De,zerosLike:Et},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const hS=M();hS.registerFlag("KEEP_INTERMEDIATE_TENSORS",()=>!1,e=>{e&&console.warn("Keep intermediate tensors is ON. This will print the values of all intermediate tensors during model inference. Not all models support this mode. For details, check e2e/benchmarks/ model_config.js. This significantly impacts performance.")});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * =============================================================================
 */var vt;(function(e){e[e.DT_INVALID=0]="DT_INVALID",e[e.DT_FLOAT=1]="DT_FLOAT",e[e.DT_DOUBLE=2]="DT_DOUBLE",e[e.DT_INT32=3]="DT_INT32",e[e.DT_UINT8=4]="DT_UINT8",e[e.DT_INT16=5]="DT_INT16",e[e.DT_INT8=6]="DT_INT8",e[e.DT_STRING=7]="DT_STRING",e[e.DT_COMPLEX64=8]="DT_COMPLEX64",e[e.DT_INT64=9]="DT_INT64",e[e.DT_BOOL=10]="DT_BOOL",e[e.DT_QINT8=11]="DT_QINT8",e[e.DT_QUINT8=12]="DT_QUINT8",e[e.DT_QINT32=13]="DT_QINT32",e[e.DT_BFLOAT16=14]="DT_BFLOAT16",e[e.DT_QINT16=15]="DT_QINT16",e[e.DT_QUINT16=16]="DT_QUINT16",e[e.DT_UINT16=17]="DT_UINT16",e[e.DT_COMPLEX128=18]="DT_COMPLEX128",e[e.DT_HALF=19]="DT_HALF",e[e.DT_RESOURCE=20]="DT_RESOURCE",e[e.DT_VARIANT=21]="DT_VARIANT",e[e.DT_UINT32=22]="DT_UINT32",e[e.DT_UINT64=23]="DT_UINT64",e[e.DT_FLOAT_REF=101]="DT_FLOAT_REF",e[e.DT_DOUBLE_REF=102]="DT_DOUBLE_REF",e[e.DT_INT32_REF=103]="DT_INT32_REF",e[e.DT_UINT8_REF=104]="DT_UINT8_REF",e[e.DT_INT16_REF=105]="DT_INT16_REF",e[e.DT_INT8_REF=106]="DT_INT8_REF",e[e.DT_STRING_REF=107]="DT_STRING_REF",e[e.DT_COMPLEX64_REF=108]="DT_COMPLEX64_REF",e[e.DT_INT64_REF=109]="DT_INT64_REF",e[e.DT_BOOL_REF=110]="DT_BOOL_REF",e[e.DT_QINT8_REF=111]="DT_QINT8_REF",e[e.DT_QUINT8_REF=112]="DT_QUINT8_REF",e[e.DT_QINT32_REF=113]="DT_QINT32_REF",e[e.DT_BFLOAT16_REF=114]="DT_BFLOAT16_REF",e[e.DT_QINT16_REF=115]="DT_QINT16_REF",e[e.DT_QUINT16_REF=116]="DT_QUINT16_REF",e[e.DT_UINT16_REF=117]="DT_UINT16_REF",e[e.DT_COMPLEX128_REF=118]="DT_COMPLEX128_REF",e[e.DT_HALF_REF=119]="DT_HALF_REF",e[e.DT_RESOURCE_REF=120]="DT_RESOURCE_REF",e[e.DT_VARIANT_REF=121]="DT_VARIANT_REF",e[e.DT_UINT32_REF=122]="DT_UINT32_REF",e[e.DT_UINT64_REF=123]="DT_UINT64_REF"})(vt||(vt={}));var wo;(function(e){(function(t){t[t.LEGACY=0]="LEGACY",t[t.V1=1]="V1",t[t.V2=2]="V2"})(e.CheckpointFormatVersion||(e.CheckpointFormatVersion={}))})(wo||(wo={}));/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const pS={};function pf(e){return pS[e]}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function f(e,t,n,r,s){const a=t.inputParams[e];if(a&&a.inputIndexStart!==void 0){const i=a.inputIndexStart,u=a.inputIndexEnd===0?void 0:a.inputIndexEnd===void 0?i+1:a.inputIndexEnd,l=i<0?t.inputNames.length+i:i;if(a.type==="tensor")return lt(t.inputNames[l],n,r,s);if(a.type==="tensors"){const p=t.inputs.slice(i,u);return t.inputNames.slice(i,u).filter((g,N)=>{var w;return((w=p[N])===null||w===void 0?void 0:w.op)!=="NoOp"}).map(g=>lt(g,n,r,s))}const h=lt(t.inputNames[l],n,r,s),c=h.dataSync();return a.type==="number"?c[0]:_e(h.shape,c)}const o=t.attrParams[e];return o&&o.value}function lt(e,t,n,r){const[s,a]=St(e,n);if(r!=null){const i=r.getHashTableHandleByName(s);if(i!=null)return i}const o=n.currentContextIds.find(i=>!!t[br(s,i)]);return o!==void 0?t[br(s,o)][a]:void 0}function No(e,t,n){return t[br(e,n.currentContextId)]}function Kt(e,t){const[n,r,s]=St(e,t);return[br(n,t&&t.currentContextId),r,s]}function br(e,t){return t?`${e}-${t}`:e}function St(e,t){if(e==="")return["",0,void 0];const n=t!=null&&t.parseNodeNameCache!=null;if(n){const a=t.parseNodeNameCache.get(e);if(a!=null)return a}const r=e.split(":");let s;if(r.length===1)s=[e,0,void 0];else{const a=r[0],o=r.length===3?r[1]:void 0,i=Number(r[r.length-1]);s=[a,i,o]}return n&&t.parseNodeNameCache.set(e,s),s}function ur(e,t,n){let r=f("pad",e,t,n);if(r==="explicit"){r=f("explicitPaddings",e,t,n);const s=[[0,0],[0,0],[0,0],[0,0]];for(let a=0;a<4;a++)s[a][0]=r[a*2],s[a][1]=r[a*2+1];return s}return r}function Xt(e){return e.kept?e:Jt(e)}/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const fS=[{tfOpName:"Add",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"AddV2",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"AddN",category:"arithmetic",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}]},{tfOpName:"BiasAdd",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"Sub",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"RealDiv",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Div",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"DivNoNan",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"FloorDiv",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Mul",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Maximum",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Minimum",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Pow",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"SquaredDifference",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Mod",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"FloorMod",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}],dS=Object.freeze(Object.defineProperty({__proto__:null,json:fS},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const mS=[{tfOpName:"Abs",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Acos",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Asin",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Atan",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Atan2",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"y",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Ceil",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ClipByValue",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"clipValueMin",type:"number"},{start:2,name:"clipValueMax",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Complex",category:"basic_math",inputs:[{start:0,name:"real",type:"tensor"},{start:1,name:"imag",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ComplexAbs",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Cos",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Cosh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Elu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Exp",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Floor",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Log",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Imag",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"Tout",name:"outputType",type:"dtype",notSupported:!0}]},{tfOpName:"Neg",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Real",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"Tout",name:"outputType",type:"dtype",notSupported:!0}]},{tfOpName:"Prelu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"alpha",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Relu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Relu6",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Selu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sigmoid",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sin",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sinh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sqrt",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Rsqrt",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Square",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Tan",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Tanh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sign",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Round",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Expm1",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Log1p",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Reciprocal",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Softplus",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Asinh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Acosh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Atanh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Erf",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LeakyRelu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"alpha",name:"alpha",type:"number",defaultValue:.2},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"IsNan",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"IsFinite",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"IsInf",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}],gS=Object.freeze(Object.defineProperty({__proto__:null,json:mS},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const yS=[{tfOpName:"EmptyTensorList",category:"control",inputs:[{start:0,name:"elementShape",type:"shape"},{start:1,name:"maxNumElements",type:"number"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"LoopCond",category:"control",inputs:[{start:0,name:"pred",type:"tensor"}]},{tfOpName:"Switch",category:"control",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"pred",type:"tensor"}]},{tfOpName:"Merge",category:"control",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}]},{tfOpName:"Enter",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"frame_name",name:"frameName",type:"string"},{tfName:"is_constant",name:"isConstant",type:"bool"}]},{tfOpName:"Exit",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"NextIteration",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"TensorArrayV3",category:"control",inputs:[{start:0,name:"size",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"element_shape",name:"elementShape",type:"shape"},{tfName:"dynamic_size",name:"dynamicSize",type:"bool"},{tfName:"clear_after_read",name:"clearAfterRead",type:"bool"},{tfName:"identical_element_shapes",name:"identicalElementShapes",type:"bool"},{tfName:"tensor_array_name",name:"name",type:"string"}]},{tfOpName:"TensorArrayWriteV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"tensor",type:"tensor"},{start:3,name:"flowIn",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"TensorArrayReadV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"flowIn",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"TensorArrayGatherV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"flowIn",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"element_shape",name:"elementShape",type:"shape"}]},{tfOpName:"TensorArrayScatterV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"tensor",type:"tensor"},{start:3,name:"flowIn",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"TensorArrayConcatV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"flowIn",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"element_shape_except0",name:"elementShapeExcept0",type:"shape",notSupported:!0}]},{tfOpName:"TensorArraySplitV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"tensor",type:"tensor"},{start:2,name:"lengths",type:"number[]"},{start:3,name:"flowIn",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"TensorArraySizeV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"flowIn",type:"number"}]},{tfOpName:"TensorArrayCloseV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"}]},{tfOpName:"StatelessIf",category:"control",inputs:[{start:0,name:"cond",type:"tensor"},{start:1,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"then_branch",name:"thenBranch",type:"func"},{tfName:"else_branch",name:"elseBranch",type:"func"}]},{tfOpName:"If",category:"control",inputs:[{start:0,name:"cond",type:"tensor"},{start:1,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"then_branch",name:"thenBranch",type:"func"},{tfName:"else_branch",name:"elseBranch",type:"func"}]},{tfOpName:"StatelessWhile",category:"control",inputs:[{start:0,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"cond",name:"cond",type:"func"},{tfName:"body",name:"body",type:"func"}]},{tfOpName:"While",category:"control",inputs:[{start:0,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"cond",name:"cond",type:"func"},{tfName:"body",name:"body",type:"func"}]},{tfOpName:"TensorListScatter",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListScatterV2",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"elementShape",type:"shape"},{start:3,name:"numElements",type:"number"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListGather",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListGetItem",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListSetItem",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"tensor",type:"tensor"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListReserve",category:"control",inputs:[{start:0,name:"elementShape",type:"shape"},{start:1,name:"numElements",type:"number"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListFromTensor",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListStack",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"},{tfName:"num_elements",name:"numElements",type:"dtype"}]},{tfOpName:"TensorListSplit",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"elementShape",type:"shape"},{start:2,name:"lengths",type:"number[]"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListConcat",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"}],attrs:[{tfName:"element_shape",name:"elementShape",type:"shape"},{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListConcatV2",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"}],attrs:[{tfName:"element_shape",name:"elementShape",type:"shape"},{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListPopBack",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListPushBack",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"tensor",type:"tensor"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListLength",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"}]},{tfOpName:"TensorListResize",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"size",type:"number"}]}],bS=Object.freeze(Object.defineProperty({__proto__:null,json:yS},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const wS=[{tfOpName:"AvgPool",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MaxPool",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[],notSupported:!0},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MaxPoolWithArgmax",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"include_batch_in_index",name:"includeBatchInIndex",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"AvgPool3D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MaxPool3D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Conv1D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"stride",name:"stride",type:"number"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NWC"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"dilation",name:"dilation",type:"number",defaultValue:1}]},{tfOpName:"Conv2D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"useCudnnOnGpu",name:"useCudnnOnGpu",type:"bool"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"_FusedConv2D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"},{start:2,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"num_args",name:"numArgs",type:"number"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"use_cudnn_on_gpu",name:"useCudnnOnGpu",type:"bool",defaultValue:!0},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"dilations",name:"dilations",type:"number[]",defaultValue:[1,1,1,1]},{tfName:"fused_ops",name:"fusedOps",type:"string[]",defaultValue:[]},{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:1e-4},{tfName:"leakyrelu_alpha",name:"leakyreluAlpha",type:"number",defaultValue:.2}]},{tfOpName:"Conv2DBackpropInput",category:"convolution",inputs:[{start:2,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"},{start:0,name:"outputShape",type:"number[]"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]",notSupported:!0}]},{tfOpName:"DepthwiseConv2d",category:"convolution",inputs:[{start:0,name:"input",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"DepthwiseConv2dNative",category:"convolution",inputs:[{start:0,name:"input",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"FusedDepthwiseConv2dNative",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"},{start:2,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"num_args",name:"numArgs",type:"number"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"dilations",name:"dilations",type:"number[]",defaultValue:[1,1,1,1]},{tfName:"fused_ops",name:"fusedOps",type:"string[]",defaultValue:[]},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]}]},{tfOpName:"Conv3D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"Dilation2D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"rates",name:"dilations",type:"number[]"},{tfName:"padding",name:"pad",type:"string"}]}],NS=Object.freeze(Object.defineProperty({__proto__:null,json:wS},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const vS=[{tfOpName:"Fill",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"},{start:1,name:"value",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"LinSpace",category:"creation",inputs:[{start:0,name:"start",type:"number"},{start:1,name:"stop",type:"number"},{start:2,name:"num",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"OneHot",category:"creation",inputs:[{start:0,name:"indices",type:"tensor"},{start:1,name:"depth",type:"number"},{start:2,name:"onValue",type:"number",defaultValue:1},{start:3,name:"offValue",type:"number",defaultValue:0}],attrs:[{tfName:"axis",name:"axis",type:"number",notSupported:!0},{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"Ones",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"OnesLike",category:"creation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"}]},{tfOpName:"RandomStandardNormal",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"seed",name:"seed",type:"number",defaultValue:0},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0},{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"T",name:"T",type:"number",notSupported:!0}]},{tfOpName:"RandomUniform",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"minval",name:"minval",type:"number",defaultValue:0},{tfName:"maxval",name:"maxval",type:"number",defaultValue:1},{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"seed",name:"seed",type:"number",defaultValue:0},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0},{tfName:"T",name:"T",type:"number",notSupported:!0}]},{tfOpName:"RandomUniformInt",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"minval",name:"minval",type:"number"},{tfName:"maxval",name:"maxval",type:"number"},{tfName:"seed",name:"seed",type:"number",defaultValue:0},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0}]},{tfOpName:"Range",category:"creation",inputs:[{start:0,name:"start",type:"number"},{start:1,name:"stop",type:"number"},{start:2,name:"step",type:"number",defaultValue:0}],attrs:[{tfName:"Tidx",name:"dtype",type:"dtype"}]},{tfOpName:"TruncatedNormal",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"means",name:"mean",type:"number",defaultValue:0},{tfName:"stddev",name:"stdDev",type:"number",defaultValue:1},{tfName:"seed",name:"seed",type:"number"},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0},{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"T",name:"T",type:"number",notSupported:!0}]},{tfOpName:"Zeros",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"ZerosLike",category:"creation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"Multinomial",category:"creation",inputs:[{start:0,name:"logits",type:"tensor"},{start:1,name:"numSamples",type:"number"}],attrs:[{tfName:"seed",name:"seed",type:"number"},{tfName:"seed2",name:"seed2",type:"number"},{tfName:"T",name:"dtype",type:"dtype"},{tfName:"output_dtype",name:"output_dtype",type:"dtype"}]}],SS=Object.freeze(Object.defineProperty({__proto__:null,json:vS},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const TS=[{tfOpName:"NonMaxSuppressionV2",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"}]},{tfOpName:"NonMaxSuppressionV3",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"},{start:4,name:"scoreThreshold",type:"number"}]},{tfOpName:"NonMaxSuppressionV4",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"},{start:4,name:"scoreThreshold",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"T_threshold",name:"threshold",type:"dtype",notSupported:!0},{tfName:"pad_to_max_output_size",name:"padToMaxOutputSize",type:"bool"}]},{tfOpName:"NonMaxSuppressionV5",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"},{start:4,name:"scoreThreshold",type:"number"},{start:5,name:"softNmsSigma",type:"number"}]},{tfOpName:"Where",category:"dynamic",inputs:[{start:0,name:"condition",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ListDiff",category:"dynamic",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"y",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}],ES=Object.freeze(Object.defineProperty({__proto__:null,json:TS},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const $S=[{tfOpName:"LowerBound",category:"evaluation",inputs:[{start:0,name:"sortedSequence",type:"tensor"},{start:1,name:"values",type:"tensor"}]},{tfOpName:"TopKV2",category:"evaluation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"k",type:"number"}],attrs:[{tfName:"sorted",name:"sorted",type:"bool"}]},{tfOpName:"UpperBound",category:"evaluation",inputs:[{start:0,name:"sortedSequence",type:"tensor"},{start:1,name:"values",type:"tensor"}]},{tfOpName:"Unique",category:"evaluation",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"UniqueV2",category:"evaluation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]}],_S=Object.freeze(Object.defineProperty({__proto__:null,json:$S},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const kS=[{tfOpName:"PlaceholderWithDefault",category:"graph",inputs:[{start:0,name:"default",type:"tensor"}],attrs:[{tfName:"shape",name:"shape",type:"shape"},{tfName:"dtype",name:"dtype",type:"dtype"}]},{tfOpName:"Placeholder",category:"graph",attrs:[{tfName:"shape",name:"shape",type:"shape"},{tfName:"dtype",name:"dtype",type:"dtype"}]},{tfOpName:"Const",category:"graph"},{tfOpName:"Identity",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"IdentityN",category:"graph",inputs:[{start:0,end:0,name:"x",type:"tensors"}]},{tfOpName:"Snapshot",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"Rank",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"Size",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"Shape",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"ShapeN",category:"graph",inputs:[{start:0,end:0,name:"x",type:"tensors"}]},{tfOpName:"Print",category:"graph",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"data",type:"tensors"}],attrs:[{tfName:"message",name:"message",type:"string"},{tfName:"first_n",name:"firstN",type:"number",notSupported:!0},{tfName:"summarize",name:"summarize",type:"number",defaultValue:3}]},{tfOpName:"NoOp",category:"graph",inputs:[]},{tfOpName:"StopGradient",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"FakeQuantWithMinMaxVars",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"min",name:"min",type:"number"},{tfName:"max",name:"max",type:"number"}]}],IS=Object.freeze(Object.defineProperty({__proto__:null,json:kS},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const xS=[{tfOpName:"HashTable",category:"hash_table",inputs:[],attrs:[{tfName:"shared_name",name:"sharedName",type:"string"},{tfName:"use_node_name_sharing",name:"useNodeNameSharing",type:"bool"},{tfName:"key_dtype",name:"keyDType",type:"dtype"},{tfName:"value_dtype",name:"valueDType",type:"dtype"}]},{tfOpName:"HashTableV2",category:"hash_table",inputs:[],attrs:[{tfName:"shared_name",name:"sharedName",type:"string"},{tfName:"use_node_name_sharing",name:"useNodeNameSharing",type:"bool"},{tfName:"key_dtype",name:"keyDType",type:"dtype"},{tfName:"value_dtype",name:"valueDType",type:"dtype"}]},{tfOpName:"LookupTableImport",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableImportV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableFind",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableFindV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableSize",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"}]},{tfOpName:"LookupTableSizeV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"}]},{tfOpName:"InitializeTable",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}]},{tfOpName:"InitializeTableV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}]}],AS=Object.freeze(Object.defineProperty({__proto__:null,json:xS},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const OS=[{tfOpName:"ResizeBilinear",category:"image",inputs:[{start:0,name:"images",type:"tensor"},{start:1,name:"size",type:"number[]"}],attrs:[{tfName:"align_corners",name:"alignCorners",type:"bool"},{tfName:"half_pixel_centers",name:"halfPixelCenters",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ResizeNearestNeighbor",category:"image",inputs:[{start:0,name:"images",type:"tensor"},{start:1,name:"size",type:"number[]"}],attrs:[{tfName:"align_corners",name:"alignCorners",type:"bool"},{tfName:"half_pixel_centers",name:"halfPixelCenters",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"CropAndResize",category:"image",inputs:[{start:0,name:"image",type:"tensor"},{start:1,name:"boxes",type:"tensor"},{start:2,name:"boxInd",type:"tensor"},{start:3,name:"cropSize",type:"number[]"}],attrs:[{tfName:"method",name:"method",type:"string"},{tfName:"extrapolation_value",name:"extrapolationValue",type:"number"}]},{tfOpName:"ImageProjectiveTransformV3",category:"image",inputs:[{start:0,name:"images",type:"tensor"},{start:1,name:"transforms",type:"tensor"},{start:2,name:"outputShape",type:"number[]"},{start:3,name:"fillValue",type:"number"}],attrs:[{tfName:"interpolation",name:"interpolation",type:"string"},{tfName:"fill_mode",name:"fillMode",type:"string"}]}],DS=Object.freeze(Object.defineProperty({__proto__:null,json:OS},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const FS=[{tfOpName:"Equal",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"NotEqual",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Greater",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"GreaterEqual",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Less",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LessEqual",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LogicalAnd",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LogicalNot",category:"logical",inputs:[{start:0,name:"a",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LogicalOr",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Select",category:"logical",inputs:[{start:0,name:"condition",type:"tensor"},{start:1,name:"a",type:"tensor"},{start:2,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"SelectV2",category:"logical",inputs:[{start:0,name:"condition",type:"tensor"},{start:1,name:"a",type:"tensor"},{start:2,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"BitwiseAnd",category:"logical",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"y",type:"tensor"}]}],RS=Object.freeze(Object.defineProperty({__proto__:null,json:FS},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const BS=[{tfOpName:"_FusedMatMul",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"},{start:2,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"num_args",name:"numArgs",type:"number"},{tfName:"fused_ops",name:"fusedOps",type:"string[]",defaultValue:[]},{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:1e-4},{tfName:"transpose_a",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"transpose_b",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"leakyrelu_alpha",name:"leakyreluAlpha",type:"number",defaultValue:.2},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MatMul",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"transpose_a",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"transpose_b",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"BatchMatMul",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"adj_x",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"adj_y",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"BatchMatMulV2",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"adj_x",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"adj_y",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Transpose",category:"matrices",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"perm",type:"number[]"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Einsum",category:"matrices",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}],attrs:[{tfName:"equation",name:"equation",type:"string"},{tfName:"N",name:"n",type:"number",defaultValue:2},{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"MatrixBandPart",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"numLower",type:"tensor"},{start:1,name:"numUpper",type:"tensor"}]}],PS=Object.freeze(Object.defineProperty({__proto__:null,json:BS},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const CS=[{tfOpName:"EuclideanNorm",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool",defaultValue:!1}]},{tfOpName:"FusedBatchNorm",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"scale",type:"tensor"},{start:2,name:"offset",type:"tensor"},{start:3,name:"mean",type:"tensor"},{start:4,name:"variance",type:"tensor"}],attrs:[{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:.001},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"FusedBatchNormV2",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"scale",type:"tensor"},{start:2,name:"offset",type:"tensor"},{start:3,name:"mean",type:"tensor"},{start:4,name:"variance",type:"tensor"}],attrs:[{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:.001},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"FusedBatchNormV3",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"scale",type:"tensor"},{start:2,name:"offset",type:"tensor"},{start:3,name:"mean",type:"tensor"},{start:4,name:"variance",type:"tensor"}],attrs:[{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:.001},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"LRN",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"depth_radius",name:"radius",type:"number",defaultValue:5},{tfName:"bias",name:"bias",type:"number",defaultValue:1},{tfName:"alpha",name:"alpha",type:"number",defaultValue:1},{tfName:"beta",name:"beta",type:"number",defaultValue:.5}]},{tfOpName:"Softmax",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"LogSoftmax",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"}]}],LS=Object.freeze(Object.defineProperty({__proto__:null,json:CS},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const zS=[{tfOpName:"Bincount",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"size",type:"number"},{start:2,name:"weights",type:"tensor"}]},{tfOpName:"DenseBincount",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"size",type:"number"},{start:2,name:"weights",type:"tensor"}],attrs:[{tfName:"binary_output",name:"binaryOutput",type:"bool"}]},{tfOpName:"Max",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Mean",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Min",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Sum",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"All",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Any",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"ArgMax",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]},{tfOpName:"ArgMin",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]},{tfOpName:"Prod",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Cumprod",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}],attrs:[{tfName:"exclusive",name:"exclusive",type:"bool"},{tfName:"reverse",name:"reverse",type:"bool"}]},{tfOpName:"Cumsum",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}],attrs:[{tfName:"exclusive",name:"exclusive",type:"bool"},{tfName:"reverse",name:"reverse",type:"bool"}]}],MS=Object.freeze(Object.defineProperty({__proto__:null,json:zS},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const VS=[{tfOpName:"ConcatV2",category:"slice_join",inputs:[{start:0,end:-1,name:"tensors",type:"tensors"},{start:-1,name:"axis",type:"number"}],attrs:[{tfName:"N",name:"n",type:"number",defaultValue:2}]},{tfOpName:"Concat",category:"slice_join",inputs:[{start:1,end:0,name:"tensors",type:"tensors"},{start:0,name:"axis",type:"number"}],attrs:[{tfName:"N",name:"n",type:"number",defaultValue:2}]},{tfOpName:"GatherV2",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"axis",type:"number",defaultValue:0}],attrs:[{tfName:"batch_dims",name:"batchDims",type:"number",defaultValue:0}]},{tfOpName:"Gather",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"indices",type:"tensor"}],attrs:[{tfName:"validate_indices",name:"validateIndices",type:"bool",notSupported:!0}]},{tfOpName:"Reverse",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"dims",type:"bool[]"}]},{tfOpName:"ReverseV2",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}]},{tfOpName:"Slice",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"begin",type:"number[]"},{start:2,name:"size",type:"number[]"}]},{tfOpName:"StridedSlice",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"begin",type:"number[]"},{start:2,name:"end",type:"number[]"},{start:3,name:"strides",type:"number[]"}],attrs:[{tfName:"begin_mask",name:"beginMask",type:"number",defaultValue:0},{tfName:"end_mask",name:"endMask",type:"number",defaultValue:0},{tfName:"new_axis_mask",name:"newAxisMask",type:"number",defaultValue:0},{tfName:"ellipsis_mask",name:"ellipsisMask",type:"number",defaultValue:0},{tfName:"shrink_axis_mask",name:"shrinkAxisMask",type:"number",defaultValue:0}]},{tfOpName:"Pack",category:"slice_join",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}],attrs:[{tfName:"axis",name:"axis",type:"number",defaultValue:0}]},{tfOpName:"Unpack",category:"slice_join",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"axis",name:"axis",type:"number",defaultValue:0},{tfName:"num",name:"num",type:"number",defaultValue:0,notSupported:!0}]},{tfOpName:"Tile",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"reps",type:"number[]"}]},{tfOpName:"Split",category:"slice_join",inputs:[{start:0,name:"axis",type:"number",defaultValue:0},{start:1,name:"x",type:"tensor"}],attrs:[{tfName:"num_split",name:"numOrSizeSplits",type:"number",defaultValue:1}]},{tfOpName:"SplitV",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"numOrSizeSplits",type:"number[]"},{start:2,name:"axis",type:"number",defaultValue:0}]},{tfOpName:"ScatterNd",category:"slice_join",inputs:[{start:0,name:"indices",type:"tensor"},{start:1,name:"values",type:"tensor"},{start:2,name:"shape",type:"number[]"}]},{tfOpName:"GatherNd",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"indices",type:"tensor"}]},{tfOpName:"SparseToDense",category:"slice_join",inputs:[{start:0,name:"sparseIndices",type:"tensor"},{start:1,name:"outputShape",type:"number[]"},{start:2,name:"sparseValues",type:"tensor"},{start:3,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"validate_indices",name:"validateIndices",type:"bool",defaultValue:!1,notSupported:!0}]},{tfOpName:"TensorScatterUpdate",category:"slice_join",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"values",type:"tensor"}]}],WS=Object.freeze(Object.defineProperty({__proto__:null,json:VS},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const US=[{tfOpName:"SparseFillEmptyRows",category:"sparse",inputs:[{start:0,name:"indices",type:"tensor"},{start:1,name:"values",type:"tensor"},{start:2,name:"denseShape",type:"tensor"},{start:3,name:"defaultValue",type:"tensor"}]},{tfOpName:"SparseReshape",category:"sparse",inputs:[{start:0,name:"inputIndices",type:"tensor"},{start:1,name:"inputShape",type:"tensor"},{start:2,name:"newShape",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"SparseSegmentMean",category:"sparse",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"segmentIds",type:"tensor"}]},{tfOpName:"SparseSegmentSum",category:"sparse",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"segmentIds",type:"tensor"}]}],qS=Object.freeze(Object.defineProperty({__proto__:null,json:US},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const HS=[{tfOpName:"FFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"IFFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"RFFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"fft_length",type:"number",notSupported:!0}]},{tfOpName:"IRFFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"fft_length",type:"number",notSupported:!0}]}],jS=Object.freeze(Object.defineProperty({__proto__:null,json:HS},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const GS=[{tfOpName:"StaticRegexReplace",category:"string",inputs:[{start:0,name:"input",type:"tensor"}],attrs:[{tfName:"pattern",name:"pattern",type:"string"},{tfName:"rewrite",name:"rewrite",type:"string"},{tfName:"replace_global",name:"replaceGlobal",type:"bool"}]},{tfOpName:"StringNGrams",category:"string",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"dataSplits",type:"tensor"}],attrs:[{tfName:"separator",name:"separator",type:"string"},{tfName:"ngram_widths",name:"nGramWidths",type:"number[]"},{tfName:"left_pad",name:"leftPad",type:"string"},{tfName:"right_pad",name:"rightPad",type:"string"},{tfName:"pad_width",name:"padWidth",type:"number"},{tfName:"preserve_short_sequences",name:"preserveShortSequences",type:"bool"}],outputs:["ngrams","ngrams_splits"]},{tfOpName:"StringSplit",category:"string",inputs:[{start:0,name:"input",type:"tensor"},{start:1,name:"delimiter",type:"tensor"}],attrs:[{tfName:"skip_empty",name:"skipEmpty",type:"bool"}],outputs:["indices","values","shape"]},{tfOpName:"StringToHashBucketFast",category:"string",inputs:[{start:0,name:"input",type:"tensor"}],attrs:[{tfName:"num_buckets",name:"numBuckets",type:"number"}]}],KS=Object.freeze(Object.defineProperty({__proto__:null,json:GS},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const XS=[{tfOpName:"Cast",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"SrcT",name:"sdtype",type:"dtype",notSupported:!0},{tfName:"DstT",name:"dtype",type:"dtype"}]},{tfOpName:"ExpandDims",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]},{tfOpName:"MirrorPad",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"padding",type:"number[]"}],attrs:[{tfName:"mode",name:"mode",type:"string"}]},{tfOpName:"Pad",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"padding",type:"number[]"}],attrs:[{tfName:"constant_value",name:"constantValue",type:"number",defaultValue:0}]},{tfOpName:"PadV2",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"padding",type:"number[]"},{start:2,name:"constantValue",type:"number",defaultValue:0}]},{tfOpName:"Reshape",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"shape",type:"number[]"}]},{tfOpName:"EnsureShape",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"shape",type:"number[]"}]},{tfOpName:"Squeeze",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"axis",tfDeprecatedName:"squeeze_dims",name:"axis",type:"number[]"}]},{tfOpName:"SpaceToBatchND",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"blockShape",type:"number[]"},{start:2,name:"paddings",type:"number[]"}]},{tfOpName:"BatchToSpaceND",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"blockShape",type:"number[]"},{start:2,name:"crops",type:"number[]"}]},{tfOpName:"DepthToSpace",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"block_size",name:"blockSize",type:"number"},{tfName:"data_format",name:"dataFormat",type:"string"}]},{tfOpName:"BroadcastTo",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"shape",type:"number[]"}],attrs:[]},{tfOpName:"BroadcastArgs",category:"transformation",inputs:[{start:0,name:"s0",type:"tensor"},{start:1,name:"s1",type:"tensor"}],attrs:[]}],YS=Object.freeze(Object.defineProperty({__proto__:null,json:XS},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class vo{static get Instance(){return this._instance||(this._instance=new this)}constructor(){const t=[dS,gS,bS,NS,SS,ES,_S,IS,AS,DS,RS,PS,LS,MS,WS,qS,jS,KS,YS],n=[].concat(...t.map(r=>r.json));this.opMappers=n.reduce((r,s)=>(r[s.tfOpName]=s,r),{})}transformGraph(t,n={}){const r=t.node,s=[],a=[],o=[],i=r.reduce((N,w)=>(N[w.name]=this.mapNode(w),w.op.startsWith("Placeholder")?s.push(N[w.name]):w.op==="Const"?a.push(N[w.name]):(w.input==null||w.input.length===0)&&o.push(N[w.name]),N),{});let u=[];const l=[];let h={},c={};n!=null&&(h=this.mapSignatureEntries(n.inputs),c=this.mapSignatureEntries(n.outputs));const p=Object.keys(i);p.forEach(N=>{const w=i[N];w.inputNames.forEach((T,x)=>{const[$,,E]=Kt(T),I=i[$];if(I.outputs!=null){const A=I.outputs.indexOf(E);if(A!==-1){const F=`${$}:${A}`;w.inputNames[x]=F}}w.inputs.push(I),I.children.push(w)})}),Object.keys(c).length===0?p.forEach(N=>{const w=i[N];w.children.length===0&&l.push(w)}):Object.keys(c).forEach(N=>{const[w]=Kt(N),T=i[w];T!=null&&(T.signatureKey=c[N],l.push(T))}),Object.keys(h).length>0?Object.keys(h).forEach(N=>{const[w]=Kt(N),T=i[w];T&&(T.signatureKey=h[N],u.push(T))}):u=s;let d={};t.library!=null&&t.library.function!=null&&(d=t.library.function.reduce((N,w)=>(N[w.signature.name]=this.mapFunction(w),N),{}));const g={nodes:i,inputs:u,outputs:l,weights:a,placeholders:s,signature:n,functions:d};return o.length>0&&(g.initNodes=o),g}mapSignatureEntries(t){return Object.keys(t||{}).reduce((n,r)=>(n[t[r].name]=r,n),{})}mapNode(t){const n=pf(t.op)||this.opMappers[t.op]||{};t.attr==null&&(t.attr={});const r={name:t.name,op:t.op,category:n.category,inputNames:(t.input||[]).map(s=>s.startsWith("^")?s.slice(1):s),inputs:[],children:[],inputParams:{},attrParams:{},rawAttrs:t.attr,outputs:n.outputs};return n.inputs!=null&&(r.inputParams=n.inputs.reduce((s,a)=>(s[a.name]={type:a.type,inputIndexStart:a.start,inputIndexEnd:a.end},s),{})),n.attrs!=null&&(r.attrParams=n.attrs.reduce((s,a)=>{const o=a.type;let i;switch(a.type){case"string":i=ws(t.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=ws(t.attr,a.tfDeprecatedName,a.defaultValue));break;case"string[]":i=_s(t.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=_s(t.attr,a.tfDeprecatedName,a.defaultValue));break;case"number":i=vs(t.attr,a.tfName,a.defaultValue||0),i===void 0&&a.tfDeprecatedName&&(i=vs(t.attr,a.tfDeprecatedName,a.defaultValue));break;case"number[]":i=$s(t.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=$s(t.attr,a.tfDeprecatedName,a.defaultValue));break;case"bool":i=Ns(t.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=Ns(t.attr,a.tfDeprecatedName,a.defaultValue));break;case"bool[]":i=Is(t.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=Is(t.attr,a.tfDeprecatedName,a.defaultValue));break;case"shape":i=Es(t.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=Es(t.attr,a.tfDeprecatedName,a.defaultValue));break;case"shape[]":i=ks(t.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=ks(t.attr,a.tfDeprecatedName,a.defaultValue));break;case"dtype":i=Ss(t.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=Ss(t.attr,a.tfDeprecatedName,a.defaultValue));break;case"dtype[]":i=Ts(t.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=Ts(t.attr,a.tfDeprecatedName,a.defaultValue));break;case"func":i=So(t.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=So(t.attr,a.tfDeprecatedName,a.defaultValue));break;case"tensor":case"tensors":break;default:throw new Error(`Unsupported param type: ${a.type} for op: ${t.op}`)}return s[a.name]={value:i,type:o},s},{})),r}mapFunction(t){const n=t.nodeDef,r=[],s=[];let a={};n!=null&&(a=n.reduce((c,p)=>(c[p.name]=this.mapNode(p),p.op==="Const"&&s.push(c[p.name]),c),{}));const o=[],i=[];t.signature.inputArg.forEach(c=>{const[p]=Kt(c.name),d={name:p,op:"Placeholder",inputs:[],inputNames:[],category:"graph",inputParams:{},attrParams:{dtype:{value:Ha(c.type),type:"dtype"}},children:[]};d.signatureKey=c.name,o.push(d),a[p]=d}),Object.keys(a).forEach(c=>{const p=a[c];p.inputNames.forEach((d,g)=>{const[N,,w]=Kt(d),T=a[N];if(T.outputs!=null){const x=T.outputs.indexOf(w);if(x!==-1){const $=`${N}:${x}`;p.inputNames[g]=$}}p.inputs.push(T),T.children.push(p)})});const l=t.ret;t.signature.outputArg.forEach(c=>{const[p,d]=Kt(l[c.name]),g=a[p];g!=null&&(g.defaultOutput=d,i.push(g))});const h=this.mapArgsToSignature(t);return{nodes:a,inputs:o,outputs:i,weights:s,placeholders:r,signature:h}}mapArgsToSignature(t){return{methodName:t.signature.name,inputs:t.signature.inputArg.reduce((n,r)=>(n[r.name]=this.mapArgToTensorInfo(r),n),{}),outputs:t.signature.outputArg.reduce((n,r)=>(n[r.name]=this.mapArgToTensorInfo(r,t.ret),n),{})}}mapArgToTensorInfo(t,n){let r=t.name;return n!=null&&(r=n[r]),{name:r,dtype:t.type}}}function ZS(e){const t=M().global;if(typeof t.atob<"u")return t.atob(e);if(typeof Buffer<"u")return new Buffer(e,"base64").toString();throw new Error("Unable to decode base64 in this environment. Missing built-in atob() or Buffer()")}function ff(e,t){const n=Array.isArray(e)?String.fromCharCode.apply(null,e):ZS(e);return t?n:n.toLowerCase()}function ws(e,t,n,r=!1){const s=e[t];return s!=null?ff(s.s,r):n}function Ns(e,t,n){const r=e[t];return r?r.b:n}function vs(e,t,n){const r=e[t]||{},s=r.i!=null?r.i:r.f!=null?r.f:n;return typeof s=="number"?s:parseInt(s,10)}function Ha(e){switch(typeof e=="string"&&(e=vt[e]),e){case vt.DT_FLOAT:case vt.DT_HALF:return"float32";case vt.DT_INT32:case vt.DT_INT64:case vt.DT_INT8:case vt.DT_UINT8:return"int32";case vt.DT_BOOL:return"bool";case vt.DT_DOUBLE:return"float32";case vt.DT_STRING:return"string";case vt.DT_COMPLEX64:case vt.DT_COMPLEX128:return"complex64";default:return null}}function So(e,t,n){const r=e[t];return r&&r.func?r.func.name:n}function Ss(e,t,n){const r=e[t];return r&&r.type?Ha(r.type):n}function Ts(e,t,n){const r=e[t];return r&&r.list&&r.list.type?r.list.type.map(s=>Ha(s)):n}function df(e){if(!e.unknownRank)return e.dim!=null?e.dim.map(t=>typeof t.size=="number"?t.size:parseInt(t.size,10)):[]}function Es(e,t,n){const r=e[t];return r&&r.shape?df(r.shape):n}function $s(e,t,n){const r=e[t];return r?((r.list.f&&r.list.f.length?r.list.f:r.list.i)||[]).map(s=>typeof s=="number"?s:parseInt(s,10)):n}function _s(e,t,n,r=!1){const s=e[t];return s&&s.list&&s.list.s?s.list.s.map(a=>ff(a,r)):n}function ks(e,t,n){const r=e[t];return r&&r.list&&r.list.shape?r.list.shape.map(s=>df(s)):n}function Is(e,t,n){const r=e[t];return r&&r.list&&r.list.b?r.list.b:n}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class JS{constructor(t,n,r){this.node=t,this.tensorMap=n,this.context=r,this.inputs=[],this.attrs={},this.inputs=t.inputNames.map(s=>this.getInput(s)),t.rawAttrs!=null&&(this.attrs=Object.keys(t.rawAttrs).reduce((s,a)=>(s[a]=this.getAttr(a),s),{}))}getInput(t){return lt(t,this.tensorMap,this.context)}getAttr(t,n){const r=this.node.rawAttrs[t];if(r.tensor!=null)return lt(t,this.tensorMap,this.context);if(r.i!=null||r.f!=null)return vs(this.node.rawAttrs,t,n);if(r.s!=null)return ws(this.node.rawAttrs,t,n);if(r.b!=null)return Ns(this.node.rawAttrs,t,n);if(r.shape!=null)return Es(this.node.rawAttrs,t,n);if(r.type!=null)return Ss(this.node.rawAttrs,t,n);if(r.list!=null){if(r.list.i!=null||r.list.f!=null)return $s(this.node.rawAttrs,t,n);if(r.list.s!=null)return _s(this.node.rawAttrs,t,n);if(r.list.shape!=null)return ks(this.node.rawAttrs,t,n);if(r.list.b!=null)return Is(this.node.rawAttrs,t,n);if(r.list.type!=null)return Ts(this.node.rawAttrs,t,n)}return n}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ct=Object.freeze(Object.defineProperty({__proto__:null,OP_SCOPE_SUFFIX:Ws,abs:Tt,acos:bc,acosh:wc,add:z,addN:Nc,all:vc,any:Sc,argMax:Ys,argMin:Tc,asin:Ec,asinh:$c,atan:_c,atan2:kc,atanh:Ic,avgPool:Js,avgPool3d:Dc,basicLSTMCell:Fc,batchNorm:Dn,batchNorm2d:Rc,batchNorm3d:Bc,batchNorm4d:Pc,batchToSpaceND:Qs,bincount:ta,bitwiseAnd:Cc,booleanMaskAsync:Ep,broadcastArgs:Lc,broadcastTo:cn,buffer:qt,cast:j,ceil:zc,clipByValue:Mc,clone:Jt,complex:ee,concat:ht,concat1d:Vc,concat2d:Wc,concat3d:Uc,concat4d:qc,conv1d:Hc,conv2d:Fn,conv2dTranspose:Gc,conv3d:Kc,conv3dTranspose:Xc,cos:Yc,cosh:Zc,cosineWindow:Dr,cumprod:Jc,cumsum:Qc,denseBincount:th,depthToSpace:eh,depthwiseConv2d:Er,diag:nh,dilation2d:rh,div:Y,divNoNan:ah,dot:oh,dropout:xp,einsum:Te,elu:na,enclosingPowerOfTwo:Fa,ensureShape:ih,equal:Rn,erf:uh,euclideanNorm:hh,exp:de,expandDims:_t,expm1:ph,eye:sa,fft:xr,fill:Je,floor:aa,floorDiv:Xs,fused:Op,gather:oa,gatherND:Ip,greater:Qe,greaterEqual:ia,ifft:En,imag:Cn,image:zn,inTopKAsync:Ap,irfft:ka,isFinite:fh,isInf:dh,isNaN:mh,leakyRelu:ua,less:yr,lessEqual:$r,linalg:zp,linspace:gh,localResponseNormalization:yh,log:Ke,log1p:la,logSigmoid:wh,logSoftmax:Nh,logSumExp:ha,logicalAnd:Nn,logicalNot:pa,logicalOr:fa,logicalXor:vh,losses:Mp,lowerBound:Sh,matMul:H,max:ke,maxPool:da,maxPool3d:Th,maxPoolWithArgmax:Eh,maximum:ma,mean:vn,meshgrid:$h,min:gr,minimum:Sn,mirrorPad:_h,mod:kh,moments:Ih,movingAverage:$p,mul:B,multiRNNCell:xh,multinomial:Ah,neg:Bt,norm:Pn,notEqual:ga,oneHot:Tn,ones:ue,onesLike:Oh,op:v,outerProduct:Dh,pad:tn,pad1d:Fh,pad2d:Rh,pad3d:ya,pad4d:Bh,pool:Ph,pow:Ge,prelu:wa,print:Ks,prod:Ch,raggedGather:Lh,raggedRange:zh,raggedTensorToTensor:Mh,rand:Vh,randomGamma:qh,randomNormal:Ea,randomStandardNormal:Hh,randomUniform:Ir,randomUniformInt:jh,range:me,real:Xe,reciprocal:Gh,relu:Ln,relu6:$a,reshape:O,reverse:ge,reverse1d:Kh,reverse2d:Xh,reverse3d:Yh,reverse4d:Zh,rfft:Ar,round:_a,rsqrt:Jh,scalar:U,scatterND:_p,searchSorted:kr,selu:Qh,separableConv2d:tp,setdiff1dAsync:ep,sigmoid:Qt,sign:np,signal:Lp,sin:rp,sinh:sp,slice:X,slice1d:ap,slice2d:op,slice3d:ip,slice4d:up,softmax:lp,softplus:ca,spaceToBatchND:ba,sparse:Vp,sparseToDense:kp,spectral:Cp,split:Ye,sqrt:Ht,square:At,squaredDifference:Ia,squeeze:Mt,stack:Gt,step:xa,stridedSlice:cp,string:Wp,sub:W,sum:Q,tan:hp,tanh:mr,tensor:xt,tensor1d:kt,tensor2d:Ue,tensor3d:Aa,tensor4d:pp,tensor5d:fp,tensor6d:dp,tensorScatterUpdate:gp,tile:We,topk:yp,transpose:$n,truncatedNormal:bp,unique:wp,unsortedSegmentSum:Np,unstack:be,upperBound:vp,variable:Sp,where:te,whereAsync:Da,zeros:De,zerosLike:Et},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const QS=(e,t,n,r=ct)=>{switch(e.op){case"BiasAdd":case"AddV2":case"Add":return[r.add(f("a",e,t,n),f("b",e,t,n))];case"AddN":return[r.addN(f("tensors",e,t,n))];case"FloorMod":case"Mod":return[r.mod(f("a",e,t,n),f("b",e,t,n))];case"Mul":return[r.mul(f("a",e,t,n),f("b",e,t,n))];case"RealDiv":case"Div":return[r.div(f("a",e,t,n),f("b",e,t,n))];case"DivNoNan":return[r.divNoNan(f("a",e,t,n),f("b",e,t,n))];case"FloorDiv":return[r.floorDiv(f("a",e,t,n),f("b",e,t,n))];case"Sub":return[r.sub(f("a",e,t,n),f("b",e,t,n))];case"Minimum":return[r.minimum(f("a",e,t,n),f("b",e,t,n))];case"Maximum":return[r.maximum(f("a",e,t,n),f("b",e,t,n))];case"Pow":return[r.pow(f("a",e,t,n),f("b",e,t,n))];case"SquaredDifference":return[r.squaredDifference(f("a",e,t,n),f("b",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const tT=(e,t,n,r=ct)=>{switch(e.op){case"Abs":case"ComplexAbs":return[r.abs(f("x",e,t,n))];case"Acos":return[r.acos(f("x",e,t,n))];case"Acosh":return[r.acosh(f("x",e,t,n))];case"Asin":return[r.asin(f("x",e,t,n))];case"Asinh":return[r.asinh(f("x",e,t,n))];case"Atan":return[r.atan(f("x",e,t,n))];case"Atan2":return[r.atan2(f("x",e,t,n),f("y",e,t,n))];case"Atanh":return[r.atanh(f("x",e,t,n))];case"Ceil":return[r.ceil(f("x",e,t,n))];case"Complex":return[r.complex(f("real",e,t,n),f("imag",e,t,n))];case"Cos":return[r.cos(f("x",e,t,n))];case"Cosh":return[r.cosh(f("x",e,t,n))];case"Elu":return[r.elu(f("x",e,t,n))];case"Erf":return[r.erf(f("x",e,t,n))];case"Exp":return[r.exp(f("x",e,t,n))];case"Expm1":return[r.expm1(f("x",e,t,n))];case"Floor":return[r.floor(f("x",e,t,n))];case"Log":return[r.log(f("x",e,t,n))];case"Log1p":return[r.log1p(f("x",e,t,n))];case"Imag":return[r.imag(f("x",e,t,n))];case"Neg":return[r.neg(f("x",e,t,n))];case"Reciprocal":return[r.reciprocal(f("x",e,t,n))];case"Real":return[r.real(f("x",e,t,n))];case"Relu":return[r.relu(f("x",e,t,n))];case"Round":return[r.round(f("x",e,t,n))];case"Selu":return[r.selu(f("x",e,t,n))];case"Sigmoid":return[r.sigmoid(f("x",e,t,n))];case"Sin":return[r.sin(f("x",e,t,n))];case"Sign":return[r.sign(f("x",e,t,n))];case"Sinh":return[r.sinh(f("x",e,t,n))];case"Softplus":return[r.softplus(f("x",e,t,n))];case"Sqrt":return[r.sqrt(f("x",e,t,n))];case"Square":return[r.square(f("x",e,t,n))];case"Tanh":return[r.tanh(f("x",e,t,n))];case"Tan":return[r.tan(f("x",e,t,n))];case"ClipByValue":return[r.clipByValue(f("x",e,t,n),f("clipValueMin",e,t,n),f("clipValueMax",e,t,n))];case"Relu6":return[r.relu6(f("x",e,t,n))];case"Rsqrt":return[r.rsqrt(lt(e.inputNames[0],t,n))];case"LeakyRelu":return[r.leakyRelu(f("x",e,t,n),f("alpha",e,t,n))];case"Prelu":return[r.prelu(f("x",e,t,n),f("alpha",e,t,n))];case"IsNan":return[r.isNaN(lt(e.inputNames[0],t,n))];case"IsInf":return[r.isInf(lt(e.inputNames[0],t,n))];case"IsFinite":return[r.isFinite(lt(e.inputNames[0],t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function It(e,t,n=""){if(!(typeof e=="number"||typeof t=="number")){y(e.length===t.length,()=>n+` Shapes ${e} and ${t} must match`);for(let r=0;r<e.length;r++){const s=e[r],a=t[r];y(s<0||a<0||s===a,()=>n+` Shapes ${e} and ${t} must match`)}}}function To(e){return!(typeof e=="number"||e.some(t=>t<0))}function rn(e,t,n){let r=xs(e,n);const s=!To(r);if(s&&t.length===0)throw new Error(`Tried to calculate elements of an empty list with non-fully-defined elementShape: ${r}`);if(s&&t.forEach(a=>{r=xs(a.shape,r)}),!To(r))throw new Error(`Non-fully-defined elementShape: ${r}`);return r}function xs(e,t){if(typeof e=="number")return t;if(typeof t=="number")return e;if(e.length!==t.length)throw new Error(`Incompatible ranks during merge: ${e} vs. ${t}`);const n=[];for(let r=0;r<e.length;++r){const s=e[r],a=t[r];if(s>=0&&a>=0&&s!==a)throw new Error(`Incompatible shape during merge: ${e} vs. ${t}`);n[r]=s>=0?s:a}return n}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class eT{constructor(t,n,r,s,a,o,i){this.name=t,this.dtype=n,this.maxSize=r,this.elementShape=s,this.identicalElementShapes=a,this.dynamicSize=o,this.clearAfterRead=i,this.tensors=[],this.closed_=!1,this.idTensor=U(0),Rt(this.idTensor)}get id(){return this.idTensor.id}get closed(){return this.closed_}clearAndClose(t){this.tensors.forEach(n=>{(t==null||!t.has(n.tensor.id))&&n.tensor.dispose()}),this.tensors=[],this.closed_=!0,this.idTensor.dispose()}size(){return this.tensors.length}read(t){if(this.closed_)throw new Error(`TensorArray ${this.name} has already been closed.`);if(t<0||t>=this.size())throw new Error(`Tried to read from index ${t}, but array size is: ${this.size()}`);const n=this.tensors[t];if(n.cleared)throw new Error(`TensorArray ${this.name}: Could not read index ${t} twice because it was cleared after a previous read (perhaps try setting clear_after_read = false?).`);return this.clearAfterRead&&(n.cleared=!0),n.read=!0,n.tensor}readMany(t){return t.map(n=>this.read(n))}write(t,n){if(this.closed_)throw new Error(`TensorArray ${this.name} has already been closed.`);if(t<0||!this.dynamicSize&&t>=this.maxSize)throw new Error(`Tried to write to index ${t}, but array is not resizeable and size is: ${this.maxSize}`);const r=this.tensors[t]||{};if(n.dtype!==this.dtype)throw new Error(`TensorArray ${this.name}: Could not write to TensorArray index ${t},
          because the value dtype is ${n.dtype}, but TensorArray dtype is ${this.dtype}.`);if(this.size()===0&&(this.elementShape==null||this.elementShape.length===0)&&(this.elementShape=n.shape),It(this.elementShape,n.shape,`TensorArray ${this.name}: Could not write to TensorArray index ${t}.`),r.read)throw new Error(`TensorArray ${this.name}: Could not write to TensorArray index ${t}, because it has already been read.`);if(r.written)throw new Error(`TensorArray ${this.name}: Could not write to TensorArray index ${t}, because it has already been written.`);r.tensor=n,Rt(n),r.written=!0,this.tensors[t]=r}writeMany(t,n){if(t.length!==n.length)throw new Error(`TensorArray ${this.name}: could not write multiple tensors,because the index size: ${t.length} is not the same as tensors size: ${n.length}.`);t.forEach((r,s)=>this.write(r,n[s]))}gather(t,n){if(n&&n!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but gather requested dtype ${n}`);if(t)t=t.slice(0,this.size());else{t=[];for(let s=0;s<this.size();s++)t.push(s)}if(t.length===0)return xt([],[0].concat(this.elementShape));const r=this.readMany(t);return It(this.elementShape,r[0].shape,"TensorArray shape mismatch: "),Gt(r,0)}concat(t){if(t&&t!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but concat requested dtype ${t}`);if(this.size()===0)return xt([],[0].concat(this.elementShape));const n=[];for(let s=0;s<this.size();s++)n.push(s);const r=this.readMany(n);return It(this.elementShape,r[0].shape,`TensorArray shape mismatch: tensor array shape (${this.elementShape}) vs first tensor shape (${r[0].shape})`),ht(r,0)}scatter(t,n){if(n.dtype!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but tensor has dtype ${n.dtype}`);if(t.length!==n.shape[0])throw new Error(`Expected len(indices) == tensor.shape[0], but saw: ${t.length} vs. ${n.shape[0]}`);const r=Math.max(...t);if(!this.dynamicSize&&r>=this.maxSize)throw new Error(`Max index must be < array size (${r}  vs. ${this.maxSize})`);this.writeMany(t,be(n,0))}split(t,n){if(n.dtype!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but tensor has dtype ${n.dtype}`);let r=0;const s=t.map(u=>(r+=u,r));if(r!==n.shape[0])throw new Error(`Expected sum of lengths to be equal to
          tensor.shape[0], but sum of lengths is
        ${r}, and tensor's shape is: ${n.shape}`);if(!this.dynamicSize&&t.length!==this.maxSize)throw new Error(`TensorArray's size is not equal to the size of lengths (${this.maxSize} vs. ${t.length}), and the TensorArray is not marked as dynamically resizeable`);const a=r===0?0:n.size/r,o=[];V(()=>{n=O(n,[1,r,a]);for(let u=0;u<t.length;++u){const h=[0,u===0?0:s[u-1],0],c=[1,t[u],a];o[u]=O(X(n,h,c),this.elementShape)}return o});const i=[];for(let u=0;u<t.length;u++)i[u]=u;this.writeMany(i,o)}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Re{get id(){return this.idTensor.id}constructor(t,n,r,s=-1){this.tensors=t,this.elementShape=n,this.elementDtype=r,t!=null&&t.forEach(a=>{if(r!==a.dtype)throw new Error(`Invalid data types; op elements ${r}, but list elements ${a.dtype}`);It(n,a.shape,"TensorList shape mismatch: "),Rt(a)}),this.idTensor=U(0),this.maxNumElements=s,Rt(this.idTensor)}copy(){return new Re([...this.tensors],this.elementShape,this.elementDtype)}clearAndClose(t){this.tensors.forEach(n=>{(t==null||!t.has(n.id))&&n.dispose()}),this.tensors.length=0,this.idTensor.dispose()}size(){return this.tensors.length}stack(t,n,r=-1){if(n!==this.elementDtype)throw new Error(`Invalid data types; op elements ${n}, but list elements ${this.elementDtype}`);if(r!==-1&&this.tensors.length!==r)throw new Error(`Operation expected a list with ${r} elements but got a list with ${this.tensors.length} elements.`);It(t,this.elementShape,"TensorList shape mismatch: ");const s=rn(this.elementShape,this.tensors,t);return V(()=>{const a=this.tensors.map(o=>O(o,s));return Gt(a,0)})}popBack(t,n){if(n!==this.elementDtype)throw new Error(`Invalid data types; op elements ${n}, but list elements ${this.elementDtype}`);if(this.size()===0)throw new Error("Trying to pop from an empty list.");const r=rn(this.elementShape,this.tensors,t),s=this.tensors.pop();return s.kept=!1,It(s.shape,t,"TensorList shape mismatch: "),O(s,r)}pushBack(t){if(t.dtype!==this.elementDtype)throw new Error(`Invalid data types; op elements ${t.dtype}, but list elements ${this.elementDtype}`);if(It(t.shape,this.elementShape,"TensorList shape mismatch: "),this.maxNumElements===this.size())throw new Error("Trying to push element into a full list.");Rt(t),this.tensors.push(t)}resize(t){if(t<0)throw new Error(`TensorListResize expects size to be non-negative. Got: ${t}`);if(this.maxNumElements!==-1&&t>this.maxNumElements)throw new Error(`TensorListResize input size ${t} is greater maxNumElement ${this.maxNumElements}.`);const n=new Re([],this.elementShape,this.elementDtype,this.maxNumElements);n.tensors.length=t;for(let r=0;r<Math.min(this.tensors.length,t);++r)n.tensors[r]=this.tensors[r];return n}getItem(t,n,r){if(r!==this.elementDtype)throw new Error(`Invalid data types; op elements ${r}, but list elements ${this.elementDtype}`);if(t<0||t>this.tensors.length)throw new Error(`Trying to access element ${t} in a list with ${this.tensors.length} elements.`);if(this.tensors[t]==null)throw new Error(`element at index ${t} is null.`);It(this.tensors[t].shape,n,"TensorList shape mismatch: ");const s=rn(this.elementShape,this.tensors,n);return O(this.tensors[t],s)}setItem(t,n){if(n.dtype!==this.elementDtype)throw new Error(`Invalid data types; op elements ${n.dtype}, but list elements ${this.elementDtype}`);if(t<0||this.maxNumElements!==-1&&t>=this.maxNumElements)throw new Error(`Trying to set element ${t} in a list with max ${this.maxNumElements} elements.`);It(this.elementShape,n.shape,"TensorList shape mismatch: "),Rt(n),this.tensors[t]!=null&&(this.tensors[t].kept=!1),this.tensors[t]=n}gather(t,n,r){if(n!==this.elementDtype)throw new Error(`Invalid data types; op elements ${n}, but list elements ${this.elementDtype}`);It(this.elementShape,r,"TensorList shape mismatch: "),t=t.slice(0,this.size());const s=rn(this.elementShape,this.tensors,r);return t.length===0?xt([],[0].concat(s)):V(()=>{const a=t.map(o=>O(this.tensors[o],s));return Gt(a,0)})}concat(t,n){if(t&&t!==this.elementDtype)throw new Error(`TensorList dtype is ${this.elementDtype} but concat requested dtype ${t}`);It(this.elementShape,n,"TensorList shape mismatch: ");const r=rn(this.elementShape,this.tensors,n);return this.size()===0?xt([],[0].concat(r)):V(()=>{const s=this.tensors.map(a=>O(a,r));return ht(s,0)})}}function nT(e,t,n){const r=e.dtype;if(e.shape.length<1)throw new Error(`Tensor must be at least a vector, but saw shape: ${e.shape}`);if(e.dtype!==n)throw new Error(`Invalid data types; op elements ${e.dtype}, but list elements ${n}`);const s=e.shape.slice(1);It(s,t,"TensorList shape mismatch: ");const a=be(e);return new Re(a,t,r)}function rT(e,t,n,r){return new Re([],e,t,r)}function sT(e,t,n,r){if(t.length!==e.shape[0])throw new Error(`Expected len(indices) == tensor.shape[0], but saw: ${t.length} vs. ${e.shape[0]}`);const s=Math.max(...t);if(r!=null&&r!==-1&&s>=r)throw new Error(`Max index must be < array size (${s}  vs. ${r})`);const a=new Re([],n,e.dtype,r),o=be(e,0);return t.forEach((i,u)=>{a.setItem(i,o[u])}),a}function aT(e,t,n){let r=0;const s=t.map(h=>(r+=h,r));if(r!==e.shape[0])throw new Error(`Expected sum of lengths to be equal to
          tensor.shape[0], but sum of lengths is
        ${r}, and tensor's shape is: ${e.shape}`);const a=e.shape.slice(1),o=xs(a,n),i=r===0?0:e.size/r,u=V(()=>{const h=[];e=O(e,[1,r,i]);for(let c=0;c<t.length;++c){const d=[0,c===0?0:s[c-1],0],g=[1,t[c],i];h[c]=O(X(e,d,g),o)}return e.dispose(),h}),l=new Re([],n,e.dtype,t.length);for(let h=0;h<u.length;h++)l.setItem(h,u[h]);return l}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const oT=async(e,t,n)=>{switch(e.op){case"If":case"StatelessIf":{const r=f("thenBranch",e,t,n),s=f("elseBranch",e,t,n),a=f("cond",e,t,n),o=f("args",e,t,n);return(await a.data())[0]?n.functionMap[r].executeFunctionAsync(o,n.tensorArrayMap,n.tensorListMap):n.functionMap[s].executeFunctionAsync(o,n.tensorArrayMap,n.tensorListMap)}case"While":case"StatelessWhile":{const r=f("body",e,t,n),s=f("cond",e,t,n),a=f("args",e,t,n),o=await n.functionMap[s].executeFunctionAsync(a,n.tensorArrayMap,n.tensorListMap),i=a.map(h=>h.id);let u=await o[0].data();o.forEach(h=>{!h.kept&&i.indexOf(h.id)===-1&&h.dispose()});let l=a;for(;u[0];){const h=l;l=await n.functionMap[r].executeFunctionAsync(l,n.tensorArrayMap,n.tensorListMap);const c=l.map(d=>d.id);h.forEach(d=>{!d.kept&&i.indexOf(d.id)===-1&&c.indexOf(d.id)===-1&&d.dispose()});const p=await n.functionMap[s].executeFunctionAsync(l,n.tensorArrayMap,n.tensorListMap);u=await p[0].data(),p.forEach(d=>{!d.kept&&i.indexOf(d.id)===-1&&c.indexOf(d.id)===-1&&d.dispose()})}return l}case"LoopCond":{const r=f("pred",e,t,n);return[Xt(r)]}case"Switch":{const r=f("pred",e,t,n);let s=f("data",e,t,n);return s.kept||(s=Xt(s)),(await r.data())[0]?[void 0,s]:[s,void 0]}case"Merge":{const r=e.inputNames.find(s=>lt(s,t,n)!==void 0);if(r){const s=lt(r,t,n);return[Xt(s)]}return}case"Enter":{const r=f("frameName",e,t,n),s=f("tensor",e,t,n);return n.enterFrame(r),[Xt(s)]}case"Exit":{const r=f("tensor",e,t,n);return n.exitFrame(),[Xt(r)]}case"NextIteration":{const r=f("tensor",e,t,n);return n.nextIteration(),[Xt(r)]}case"TensorArrayV3":{const r=f("size",e,t,n),s=f("dtype",e,t,n),a=f("elementShape",e,t,n),o=f("dynamicSize",e,t,n),i=f("clearAfterRead",e,t,n),u=f("identicalElementShapes",e,t,n),l=f("name",e,t,n),h=new eT(l,s,r,a,u,o,i);return n.addTensorArray(h),[h.idTensor,U(1)]}case"TensorArrayWriteV3":{const r=f("tensorArrayId",e,t,n),s=f("index",e,t,n),a=f("tensor",e,t,n),o=n.getTensorArray(r.id);return o.write(s,a),[o.idTensor]}case"TensorArrayReadV3":{const r=f("tensorArrayId",e,t,n),s=f("index",e,t,n);return[n.getTensorArray(r.id).read(s)]}case"TensorArrayGatherV3":{const r=f("tensorArrayId",e,t,n),s=f("indices",e,t,n),a=f("dtype",e,t,n);return[n.getTensorArray(r.id).gather(s,a)]}case"TensorArrayScatterV3":{const r=f("tensorArrayId",e,t,n),s=f("indices",e,t,n),a=f("tensor",e,t,n),o=n.getTensorArray(r.id);return o.scatter(s,a),[o.idTensor]}case"TensorArrayConcatV3":{const r=f("tensorArrayId",e,t,n),s=n.getTensorArray(r.id),a=f("dtype",e,t,n);return[s.concat(a)]}case"TensorArraySplitV3":{const r=f("tensorArrayId",e,t,n),s=f("tensor",e,t,n),a=f("lengths",e,t,n),o=n.getTensorArray(r.id);return o.split(a,s),[o.idTensor]}case"TensorArraySizeV3":{const r=f("tensorArrayId",e,t,n),s=n.getTensorArray(r.id);return[U(s.size(),"int32")]}case"TensorArrayCloseV3":{const r=f("tensorArrayId",e,t,n),s=n.getTensorArray(r.id);return s.clearAndClose(),[s.idTensor]}case"TensorListSetItem":{const r=f("tensorListId",e,t,n),s=f("index",e,t,n),a=f("tensor",e,t,n),o=n.getTensorList(r.id);return o.setItem(s,a),[o.idTensor]}case"TensorListGetItem":{const r=f("tensorListId",e,t,n),s=f("index",e,t,n),a=f("elementShape",e,t,n),o=f("elementDType",e,t,n);return[n.getTensorList(r.id).getItem(s,a,o)]}case"TensorListScatterV2":case"TensorListScatter":{const r=f("indices",e,t,n),s=f("tensor",e,t,n),a=f("elementShape",e,t,n),o=f("numElements",e,t,n),i=sT(s,r,a,o);return n.addTensorList(i),[i.idTensor]}case"TensorListReserve":case"EmptyTensorList":{const r=f("elementShape",e,t,n),s=f("elementDType",e,t,n);let a;e.op==="TensorListReserve"?a="numElements":a="maxNumElements";const o=f(a,e,t,n),i=e.op==="TensorListReserve"?-1:o,u=rT(r,s,o,i);return n.addTensorList(u),[u.idTensor]}case"TensorListGather":{const r=f("tensorListId",e,t,n),s=f("indices",e,t,n),a=f("elementShape",e,t,n),o=f("elementDType",e,t,n);return[n.getTensorList(r.id).gather(s,o,a)]}case"TensorListStack":{const r=f("tensorListId",e,t,n),s=f("elementShape",e,t,n),a=f("elementDType",e,t,n),o=f("numElements",e,t,n);return[n.getTensorList(r.id).stack(s,a,o)]}case"TensorListFromTensor":{const r=f("tensor",e,t,n),s=f("elementShape",e,t,n),a=f("elementDType",e,t,n),o=nT(r,s,a);return n.addTensorList(o),[o.idTensor]}case"TensorListConcat":case"TensorListConcatV2":{const r=f("tensorListId",e,t,n),s=n.getTensorList(r.id),a=f("dtype",e,t,n),o=f("elementShape",e,t,n);return[s.concat(a,o)]}case"TensorListPushBack":{const r=f("tensorListId",e,t,n),s=f("tensor",e,t,n),a=n.getTensorList(r.id);return a.pushBack(s),[a.idTensor]}case"TensorListPopBack":{const r=f("tensorListId",e,t,n),s=f("elementShape",e,t,n),a=f("elementDType",e,t,n);return[n.getTensorList(r.id).popBack(s,a)]}case"TensorListSplit":{const r=f("tensor",e,t,n),s=f("elementShape",e,t,n),a=f("lengths",e,t,n),o=aT(r,a,s);return n.addTensorList(o),[o.idTensor]}case"TensorListLength":{const r=f("tensorListId",e,t,n),s=n.getTensorList(r.id);return[U(s.size(),"int32")]}case"TensorListResize":{const r=f("tensorListId",e,t,n),s=f("size",e,t,n),o=n.getTensorList(r.id).resize(s);return n.addTensorList(o),[o.idTensor]}default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Eo(e,t,n){const[r,s]=f("fusedOps",e,t,n),a=r==="biasadd",o=!a,i=s==="prelu",u=r==="fusedbatchnorm",l=f("numArgs",e,t,n);if(a){if(i&&l!==2)throw new Error("FusedConv2d and DepthwiseConv2d with BiasAdd and Prelu must have two extra arguments: bias and alpha.");if(!i&&a&&l!==1)throw new Error("FusedConv2d and DepthwiseConv2d with BiasAdd must have one extra argument: bias.")}if(u)throw new Error("FusedConv2d and DepthwiseConv2d with FusedBatchNorm is not supported");const h=f("strides",e,t,n),c=ur(e,t,n),p=f("dataFormat",e,t,n).toUpperCase(),d=f("dilations",e,t,n);let[g,N]=f("args",e,t,n);o&&(N=g,g=void 0);const w=f("leakyreluAlpha",e,t,n);return{stride:h,pad:c,dataFormat:p,dilations:d,biasArg:g,preluArg:N,activationFunc:s,leakyreluAlpha:w}}const iT=(e,t,n,r=ct)=>{switch(e.op){case"Conv1D":{const s=f("stride",e,t,n),a=f("pad",e,t,n),o=f("dataFormat",e,t,n).toUpperCase(),i=f("dilation",e,t,n);return[r.conv1d(f("x",e,t,n),f("filter",e,t,n),s,a,o,i)]}case"Conv2D":{const s=f("strides",e,t,n),a=ur(e,t,n),o=f("dataFormat",e,t,n).toUpperCase(),i=f("dilations",e,t,n);return[r.conv2d(f("x",e,t,n),f("filter",e,t,n),[s[1],s[2]],a,o,[i[1],i[2]])]}case"_FusedConv2D":{const{stride:s,pad:a,dataFormat:o,dilations:i,biasArg:u,preluArg:l,activationFunc:h,leakyreluAlpha:c}=Eo(e,t,n);return[r.fused.conv2d({x:f("x",e,t,n),filter:f("filter",e,t,n),strides:[s[1],s[2]],pad:a,dataFormat:o,dilations:[i[1],i[2]],bias:u,activation:h,preluActivationWeights:l,leakyreluAlpha:c})]}case"FusedDepthwiseConv2dNative":{const{stride:s,pad:a,dataFormat:o,dilations:i,biasArg:u,preluArg:l,activationFunc:h,leakyreluAlpha:c}=Eo(e,t,n);return[r.fused.depthwiseConv2d({x:f("x",e,t,n),filter:f("filter",e,t,n),strides:[s[1],s[2]],pad:a,dataFormat:o,dilations:[i[1],i[2]],bias:u,activation:h,preluActivationWeights:l,leakyreluAlpha:c})]}case"Conv2DBackpropInput":case"Conv2dTranspose":{const s=f("outputShape",e,t,n),a=f("strides",e,t,n),o=ur(e,t,n);return[r.conv2dTranspose(f("x",e,t,n),f("filter",e,t,n),s,[a[1],a[2]],o)]}case"DepthwiseConv2dNative":case"DepthwiseConv2d":{const s=f("strides",e,t,n),a=ur(e,t,n),o=f("dilations",e,t,n),i=f("dataFormat",e,t,n).toUpperCase();return[r.depthwiseConv2d(f("input",e,t,n),f("filter",e,t,n),[s[1],s[2]],a,i,[o[1],o[2]])]}case"Conv3D":{const s=f("strides",e,t,n),a=f("pad",e,t,n),o=f("dataFormat",e,t,n).toUpperCase(),i=f("dilations",e,t,n);return[r.conv3d(f("x",e,t,n),f("filter",e,t,n),[s[1],s[2],s[3]],a,o,[i[1],i[2],i[3]])]}case"AvgPool":{const s=f("strides",e,t,n),a=f("pad",e,t,n),o=f("kernelSize",e,t,n);return[r.avgPool(f("x",e,t,n),[o[1],o[2]],[s[1],s[2]],a)]}case"MaxPool":{const s=f("strides",e,t,n),a=f("pad",e,t,n),o=f("kernelSize",e,t,n);return[r.maxPool(f("x",e,t,n),[o[1],o[2]],[s[1],s[2]],a)]}case"MaxPoolWithArgmax":{const s=f("strides",e,t,n),a=f("pad",e,t,n),o=f("kernelSize",e,t,n),i=f("includeBatchInIndex",e,t,n),{result:u,indexes:l}=r.maxPoolWithArgmax(f("x",e,t,n),[o[1],o[2]],[s[1],s[2]],a,i);return[u,l]}case"AvgPool3D":{const s=f("strides",e,t,n),a=f("pad",e,t,n),o=f("kernelSize",e,t,n);return[r.avgPool3d(f("x",e,t,n),[o[1],o[2],o[3]],[s[1],s[2],s[3]],a)]}case"MaxPool3D":{const s=f("strides",e,t,n),a=f("pad",e,t,n),o=f("kernelSize",e,t,n);return[r.maxPool3d(f("x",e,t,n),[o[1],o[2],o[3]],[s[1],s[2],s[3]],a)]}case"Dilation2D":{const s=f("strides",e,t,n),a=f("pad",e,t,n),o=f("dilations",e,t,n),i=s[1],u=s[2],l=o[1],h=o[2];return[r.dilation2d(f("x",e,t,n),f("filter",e,t,n),[i,u],a,[l,h],"NHWC")]}default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const uT=(e,t,n,r=ct)=>{switch(e.op){case"Fill":{const s=f("shape",e,t,n),a=f("dtype",e,t,n),o=f("value",e,t,n);return[r.fill(s,o,a)]}case"LinSpace":{const s=f("start",e,t,n),a=f("stop",e,t,n),o=f("num",e,t,n);return[r.linspace(s,a,o)]}case"Multinomial":{const s=f("logits",e,t,n),a=f("numSamples",e,t,n),o=f("seed",e,t,n);return[r.multinomial(s,a,o)]}case"OneHot":{const s=f("indices",e,t,n),a=f("depth",e,t,n),o=f("onValue",e,t,n),i=f("offValue",e,t,n),u=f("dtype",e,t,n);return[r.oneHot(s,a,o,i,u)]}case"Ones":return[r.ones(f("shape",e,t,n),f("dtype",e,t,n))];case"OnesLike":return[r.onesLike(f("x",e,t,n))];case"RandomStandardNormal":return[r.randomStandardNormal(f("shape",e,t,n),f("dtype",e,t,n),f("seed",e,t,n))];case"RandomUniform":return[r.randomUniform(f("shape",e,t,n),f("minval",e,t,n),f("maxval",e,t,n),f("dtype",e,t,n))];case"RandomUniformInt":return[r.randomUniformInt(f("shape",e,t,n),f("minval",e,t,n),f("maxval",e,t,n),f("seed",e,t,n))];case"Range":{const s=f("start",e,t,n),a=f("stop",e,t,n),o=f("step",e,t,n);return[r.range(s,a,o,f("dtype",e,t,n))]}case"TruncatedNormal":{const s=f("shape",e,t,n),a=f("mean",e,t,n),o=f("stdDev",e,t,n),i=f("seed",e,t,n);return[r.truncatedNormal(s,a,o,f("dtype",e,t,n),i)]}case"Zeros":return[r.zeros(f("shape",e,t,n),f("dtype",e,t,n))];case"ZerosLike":return[r.zerosLike(f("x",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Gr(e,t,n){const r=f("boxes",e,t,n),s=f("scores",e,t,n),a=f("maxOutputSize",e,t,n),o=f("iouThreshold",e,t,n),i=f("scoreThreshold",e,t,n),u=f("softNmsSigma",e,t,n);return{boxes:r,scores:s,maxOutputSize:a,iouThreshold:o,scoreThreshold:i,softNmsSigma:u}}const lT=async(e,t,n,r,s=ct)=>{switch(e.op){case"NonMaxSuppressionV5":{const{boxes:a,scores:o,maxOutputSize:i,iouThreshold:u,scoreThreshold:l,softNmsSigma:h}=Gr(e,t,n),c=await s.image.nonMaxSuppressionWithScoreAsync(a,o,i,u,l,h);return[c.selectedIndices,c.selectedScores]}case"NonMaxSuppressionV4":{const{boxes:a,scores:o,maxOutputSize:i,iouThreshold:u,scoreThreshold:l}=Gr(e,t,n),h=f("padToMaxOutputSize",e,t,n),c=await s.image.nonMaxSuppressionPaddedAsync(a,o,i,u,l,h);return[c.selectedIndices,c.validOutputs]}case"NonMaxSuppressionV3":case"NonMaxSuppressionV2":{const{boxes:a,scores:o,maxOutputSize:i,iouThreshold:u,scoreThreshold:l}=Gr(e,t,n);return[await s.image.nonMaxSuppressionAsync(a,o,i,u,l)]}case"Where":{const a=s.cast(f("condition",e,t,n),"bool"),o=[await s.whereAsync(a)];return a.dispose(),o}case"ListDiff":return s.setdiff1dAsync(f("x",e,t,n),f("y",e,t,n));default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const cT=(e,t,n,r=ct)=>{switch(e.op){case"LowerBound":{const s=f("sortedSequence",e,t,n),a=f("values",e,t,n);return[r.lowerBound(s,a)]}case"TopKV2":{const s=f("x",e,t,n),a=f("k",e,t,n),o=f("sorted",e,t,n),i=r.topk(s,a,o);return[i.values,i.indices]}case"UpperBound":{const s=f("sortedSequence",e,t,n),a=f("values",e,t,n);return[r.upperBound(s,a)]}case"Unique":{const s=f("x",e,t,n),a=r.unique(s);return[a.values,a.indices]}case"UniqueV2":{const s=f("x",e,t,n),a=f("axis",e,t,n),o=r.unique(s,a);return[o.values,o.indices]}default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const hT=(e,t,n,r=ct)=>{switch(e.op){case"Const":return t[e.name];case"PlaceholderWithDefault":const s=f("default",e,t,n);return[lt(e.name,t,n)||s];case"Placeholder":return[lt(e.name,t,n)];case"Identity":case"StopGradient":case"FakeQuantWithMinMaxVars":{const h=f("x",e,t,n);return[Xt(h)]}case"IdentityN":return f("x",e,t,n).map(h=>Xt(h));case"Snapshot":const a=f("x",e,t,n);return[Xt(a)];case"Shape":return[r.tensor1d(f("x",e,t,n).shape,"int32")];case"ShapeN":return f("x",e,t,n).map(h=>r.tensor1d(h.shape));case"Size":return[r.scalar(f("x",e,t,n).size,"int32")];case"Rank":return[r.scalar(f("x",e,t,n).rank,"int32")];case"NoOp":return[r.scalar(1)];case"Print":const o=f("x",e,t,n),i=f("data",e,t,n),u=f("message",e,t,n),l=f("summarize",e,t,n);console.warn("The graph has a tf.print() operation,usually used for debugging, which slows down performance."),console.log(u);for(let h=0;h<i.length;h++)console.log(Array.prototype.slice.call(i[h].dataSync()).slice(0,l));return[o];default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class pT{get id(){return this.handle.id}constructor(t,n){this.keyDType=t,this.valueDType=n,this.handle=U(0),this.tensorMap=new Map,Rt(this.handle)}clearAndClose(){this.tensorMap.forEach(t=>t.dispose()),this.tensorMap.clear(),this.handle.dispose()}size(){return this.tensorMap.size}tensorSize(){return U(this.size(),"int32")}async import(t,n){this.checkKeyAndValueTensor(t,n);const r=await t.data();return this.tensorMap.forEach(s=>s.dispose()),this.tensorMap.clear(),V(()=>{const s=be(n),a=r.length,o=s.length;y(a===o,()=>`The number of elements doesn't match, keys has ${a} elements, the values has ${o} elements.`);for(let i=0;i<a;i++){const u=r[i],l=s[i];Rt(l),this.tensorMap.set(u,l)}return this.handle})}async find(t,n){this.checkKeyAndValueTensor(t,n);const r=await t.data();return V(()=>{const s=[];for(let a=0;a<r.length;a++){const o=r[a],i=this.findWithDefault(o,n);s.push(i)}return Gt(s)})}findWithDefault(t,n){const r=this.tensorMap.get(t);return r??n}checkKeyAndValueTensor(t,n){if(t.dtype!==this.keyDType)throw new Error(`Expect key dtype ${this.keyDType}, but got ${t.dtype}`);if(n.dtype!==this.valueDType)throw new Error(`Expect value dtype ${this.valueDType}, but got ${n.dtype}`)}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const fT=async(e,t,n,r)=>{switch(e.op){case"HashTable":case"HashTableV2":{const s=r.getHashTableHandleByName(e.name);if(s!=null)return[s];{const a=f("keyDType",e,t,n),o=f("valueDType",e,t,n),i=new pT(a,o);return r.addHashTable(e.name,i),[i.handle]}}case"InitializeTable":case"InitializeTableV2":case"LookupTableImport":case"LookupTableImportV2":{const s=f("tableHandle",e,t,n,r),a=f("keys",e,t,n),o=f("values",e,t,n);return[await r.getHashTableById(s.id).import(a,o)]}case"LookupTableFind":case"LookupTableFindV2":{const s=f("tableHandle",e,t,n,r),a=f("keys",e,t,n),o=f("defaultValue",e,t,n);return[await r.getHashTableById(s.id).find(a,o)]}case"LookupTableSize":case"LookupTableSizeV2":{const s=f("tableHandle",e,t,n,r);return[r.getHashTableById(s.id).tensorSize()]}default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const dT=(e,t,n,r=ct)=>{switch(e.op){case"ResizeBilinear":{const s=f("images",e,t,n),a=f("size",e,t,n),o=f("alignCorners",e,t,n),i=f("halfPixelCenters",e,t,n);return[r.image.resizeBilinear(s,[a[0],a[1]],o,i)]}case"ResizeNearestNeighbor":{const s=f("images",e,t,n),a=f("size",e,t,n),o=f("alignCorners",e,t,n),i=f("halfPixelCenters",e,t,n);return[r.image.resizeNearestNeighbor(s,[a[0],a[1]],o,i)]}case"CropAndResize":{const s=f("image",e,t,n),a=f("boxes",e,t,n),o=f("boxInd",e,t,n),i=f("cropSize",e,t,n),u=f("method",e,t,n),l=f("extrapolationValue",e,t,n);return[r.image.cropAndResize(s,a,o,i,u,l)]}case"ImageProjectiveTransformV3":{const s=f("images",e,t,n),a=f("transforms",e,t,n),o=f("outputShape",e,t,n),i=f("fillValue",e,t,n),u=f("interpolation",e,t,n),l=f("fillMode",e,t,n);return[r.image.transform(s,a,u.toLowerCase(),l.toLowerCase(),i,o)]}default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const mT=(e,t,n,r=ct)=>{switch(e.op){case"Equal":return[r.equal(f("a",e,t,n),f("b",e,t,n))];case"NotEqual":return[r.notEqual(f("a",e,t,n),f("b",e,t,n))];case"Greater":return[r.greater(f("a",e,t,n),f("b",e,t,n))];case"GreaterEqual":return[r.greaterEqual(f("a",e,t,n),f("b",e,t,n))];case"Less":return[r.less(f("a",e,t,n),f("b",e,t,n))];case"LessEqual":return[r.lessEqual(f("a",e,t,n),f("b",e,t,n))];case"LogicalAnd":return[r.logicalAnd(f("a",e,t,n),f("b",e,t,n))];case"LogicalNot":return[r.logicalNot(f("a",e,t,n))];case"LogicalOr":return[r.logicalOr(f("a",e,t,n),f("b",e,t,n))];case"Select":case"SelectV2":return[r.where(f("condition",e,t,n),f("a",e,t,n),f("b",e,t,n))];case"BitwiseAnd":return[r.bitwiseAnd(f("a",e,t,n),f("b",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const gT=(e,t,n,r=ct)=>{switch(e.op){case"BatchMatMul":case"BatchMatMulV2":case"MatMul":return[r.matMul(f("a",e,t,n),f("b",e,t,n),f("transposeA",e,t,n),f("transposeB",e,t,n))];case"Einsum":return[r.einsum(f("equation",e,t,n),...f("tensors",e,t,n))];case"Transpose":return[r.transpose(f("x",e,t,n),f("perm",e,t,n))];case"_FusedMatMul":const[s,a]=f("fusedOps",e,t,n),o=s==="biasadd",i=a==="prelu",u=f("numArgs",e,t,n),l=f("leakyreluAlpha",e,t,n);if(o){if(i&&u!==2)throw new Error("Fused MatMul with BiasAdd and Prelu must have two extra arguments: bias and alpha.");if(!i&&u!==1)throw new Error("Fused MatMul with BiasAdd must have one extra argument: bias.")}const[h,c]=f("args",e,t,n);return[r.fused.matMul({a:f("a",e,t,n),b:f("b",e,t,n),transposeA:f("transposeA",e,t,n),transposeB:f("transposeB",e,t,n),bias:h,activation:a,preluActivationWeights:c,leakyreluAlpha:l})];case"MatrixBandPart":return[r.linalg.bandPart(f("a",e,t,n),f("numLower",e,t,n),f("numUpper",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const yT=(e,t,n,r=ct)=>{switch(e.op){case"EuclideanNorm":return[r.euclideanNorm(f("x",e,t,n),f("axis",e,t,n),f("keepDims",e,t,n))];case"FusedBatchNorm":case"FusedBatchNormV2":return[r.batchNorm(f("x",e,t,n),f("mean",e,t,n),f("variance",e,t,n),f("offset",e,t,n),f("scale",e,t,n),f("epsilon",e,t,n))];case"FusedBatchNormV3":return[r.batchNorm(f("x",e,t,n),f("mean",e,t,n),f("variance",e,t,n),f("offset",e,t,n),f("scale",e,t,n),f("epsilon",e,t,n))];case"LRN":return[r.localResponseNormalization(f("x",e,t,n),f("radius",e,t,n),f("bias",e,t,n),f("alpha",e,t,n),f("beta",e,t,n))];case"Softmax":return[r.softmax(f("x",e,t,n))];case"LogSoftmax":return[r.logSoftmax(f("x",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const bT=(e,t,n,r=ct)=>{switch(e.op){case"RaggedGather":{const{outputNestedSplits:s,outputDenseValues:a}=r.raggedGather(f("paramsNestedSplits",e,t,n),f("paramsDenseValues",e,t,n),f("indices",e,t,n),f("outputRaggedRank",e,t,n));return s.concat(a)}case"RaggedRange":{const{rtNestedSplits:s,rtDenseValues:a}=r.raggedRange(f("starts",e,t,n),f("limits",e,t,n),f("splits",e,t,n));return[s,a]}case"RaggedTensorToTensor":return[r.raggedTensorToTensor(f("shape",e,t,n),f("values",e,t,n),f("defaultValue",e,t,n),f("rowPartitionTensors",e,t,n),f("rowPartitionTypes",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const wT=(e,t,n,r=ct)=>{switch(e.op){case"Max":{const i=f("axis",e,t,n),u=f("keepDims",e,t,n);return[r.max(f("x",e,t,n),i,u)]}case"Mean":{const i=f("axis",e,t,n),u=f("keepDims",e,t,n);return[r.mean(f("x",e,t,n),i,u)]}case"Min":{const i=f("axis",e,t,n),u=f("keepDims",e,t,n);return[r.min(f("x",e,t,n),i,u)]}case"Sum":{const i=f("axis",e,t,n),u=f("keepDims",e,t,n);return[r.sum(f("x",e,t,n),i,u)]}case"All":{const i=f("axis",e,t,n),u=f("keepDims",e,t,n);return[r.all(f("x",e,t,n),i,u)]}case"Any":{const i=f("axis",e,t,n),u=f("keepDims",e,t,n);return[r.any(f("x",e,t,n),i,u)]}case"ArgMax":{const i=f("axis",e,t,n);return[r.argMax(f("x",e,t,n),i)]}case"ArgMin":{const i=f("axis",e,t,n);return[r.argMin(f("x",e,t,n),i)]}case"Prod":{const i=f("axis",e,t,n),u=f("keepDims",e,t,n);return[r.prod(f("x",e,t,n),i,u)]}case"Cumprod":{const i=f("axis",e,t,n),u=f("exclusive",e,t,n),l=f("reverse",e,t,n);return[r.cumprod(f("x",e,t,n),i,u,l)]}case"Cumsum":{const i=f("axis",e,t,n),u=f("exclusive",e,t,n),l=f("reverse",e,t,n);return[r.cumsum(f("x",e,t,n),i,u,l)]}case"Bincount":const s=f("x",e,t,n),a=f("weights",e,t,n),o=f("size",e,t,n);return[r.bincount(s,a,o)];case"DenseBincount":{const i=f("x",e,t,n),u=f("weights",e,t,n),l=f("size",e,t,n),h=f("binaryOutput",e,t,n);return[r.denseBincount(i,u,l,h)]}default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const NT=(e,t,n,r=ct)=>{switch(e.op){case"ConcatV2":case"Concat":{const s=f("n",e,t,n),a=f("axis",e,t,n);let o=f("tensors",e,t,n);return o=o.slice(0,s),[r.concat(o,a)]}case"Gather":{const s=f("x",e,t,n),a=f("indices",e,t,n);return[r.gather(s,r.cast(a,"int32"),0)]}case"GatherV2":{const s=f("axis",e,t,n),a=f("batchDims",e,t,n),o=f("x",e,t,n),i=f("indices",e,t,n);return[r.gather(o,r.cast(i,"int32"),s,a)]}case"Reverse":{const s=f("dims",e,t,n),a=[];for(let i=0;i<s.length;i++)s[i]&&a.push(i);const o=f("x",e,t,n);return[r.reverse(o,a)]}case"ReverseV2":{const s=f("axis",e,t,n),a=f("x",e,t,n);return[r.reverse(a,s)]}case"Slice":{const s=f("begin",e,t,n),a=f("size",e,t,n);return[r.slice(f("x",e,t,n),s,a)]}case"StridedSlice":{const s=f("begin",e,t,n),a=f("end",e,t,n),o=f("strides",e,t,n),i=f("beginMask",e,t,n),u=f("endMask",e,t,n),l=f("ellipsisMask",e,t,n),h=f("newAxisMask",e,t,n),c=f("shrinkAxisMask",e,t,n),p=f("x",e,t,n);return[r.stridedSlice(p,s,a,o,i,u,l,h,c)]}case"Pack":return V(()=>{const s=f("axis",e,t,n),a=f("tensors",e,t,n),o=a[0].shape,i=r.squeeze(a[0]).shape,u=a.map(l=>{const h=Pt(l.shape,o);if(!h&&!Pt(r.squeeze(l).shape,i))throw new Error("the input tensors shape does not match");return h?l:r.reshape(l,o)});return[r.stack(u,s)]});case"Unpack":{const s=f("axis",e,t,n),a=f("tensor",e,t,n);return r.unstack(a,s)}case"Tile":{const s=f("reps",e,t,n);return[r.tile(f("x",e,t,n),s)]}case"Split":case"SplitV":{const s=f("axis",e,t,n),a=f("numOrSizeSplits",e,t,n),o=f("x",e,t,n);return r.split(o,a,s)}case"ScatterNd":{const s=f("indices",e,t,n),a=f("values",e,t,n),o=f("shape",e,t,n);return[r.scatterND(s,a,o)]}case"GatherNd":{const s=f("x",e,t,n),a=f("indices",e,t,n);return[r.gatherND(s,a)]}case"SparseToDense":{const s=f("sparseIndices",e,t,n),a=f("outputShape",e,t,n),o=f("sparseValues",e,t,n),i=f("defaultValue",e,t,n);return[r.sparseToDense(s,o,a,o.dtype===i.dtype?i:r.cast(i,o.dtype))]}case"TensorScatterUpdate":{const s=f("indices",e,t,n),a=f("values",e,t,n),o=f("tensor",e,t,n);return[r.tensorScatterUpdate(o,s,a)]}default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const vT=(e,t,n,r=ct)=>{switch(e.op){case"SparseFillEmptyRows":{const{outputIndices:s,outputValues:a,emptyRowIndicator:o,reverseIndexMap:i}=r.sparse.sparseFillEmptyRows(f("indices",e,t,n),f("values",e,t,n),f("denseShape",e,t,n),f("defaultValue",e,t,n));return[s,a,o,i]}case"SparseReshape":{const{outputIndices:s,outputShape:a}=r.sparse.sparseReshape(f("inputIndices",e,t,n),f("inputShape",e,t,n),f("newShape",e,t,n));return[s,a]}case"SparseSegmentMean":return[r.sparse.sparseSegmentMean(f("data",e,t,n),f("indices",e,t,n),f("segmentIds",e,t,n))];case"SparseSegmentSum":return[r.sparse.sparseSegmentSum(f("data",e,t,n),f("indices",e,t,n),f("segmentIds",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ST=(e,t,n,r=ct)=>{switch(e.op){case"FFT":return[r.fft(f("x",e,t,n))];case"IFFT":return[r.ifft(f("x",e,t,n))];case"RFFT":return[r.rfft(f("x",e,t,n))];case"IRFFT":return[r.irfft(f("x",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const TT=(e,t,n,r=ct)=>{switch(e.op){case"StaticRegexReplace":return[r.string.staticRegexReplace(f("input",e,t,n),f("pattern",e,t,n),f("rewrite",e,t,n),f("replaceGlobal",e,t,n))];case"StringNGrams":{const{nGrams:s,nGramsSplits:a}=r.string.stringNGrams(f("data",e,t,n),f("dataSplits",e,t,n),f("separator",e,t,n),f("nGramWidths",e,t,n),f("leftPad",e,t,n),f("rightPad",e,t,n),f("padWidth",e,t,n),f("preserveShortSequences",e,t,n));return[s,a]}case"StringSplit":{const{indices:s,values:a,shape:o}=r.string.stringSplit(f("input",e,t,n),f("delimiter",e,t,n),f("skipEmpty",e,t,n));return[s,a,o]}case"StringToHashBucketFast":return[r.string.stringToHashBucketFast(f("input",e,t,n),f("numBuckets",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ET=(e,t,n,r=ct)=>{switch(e.op){case"Cast":return[r.cast(f("x",e,t,n),f("dtype",e,t,n))];case"ExpandDims":{const s=f("axis",e,t,n);return[r.expandDims(f("x",e,t,n),s)]}case"Squeeze":{const s=f("axis",e,t,n);return[r.squeeze(f("x",e,t,n),s)]}case"Reshape":return[r.reshape(f("x",e,t,n),f("shape",e,t,n))];case"EnsureShape":return[r.ensureShape(f("x",e,t,n),f("shape",e,t,n))];case"MirrorPad":return[r.mirrorPad(f("x",e,t,n),f("padding",e,t,n),f("mode",e,t,n))];case"PadV2":case"Pad":return[r.pad(f("x",e,t,n),f("padding",e,t,n),f("constantValue",e,t,n))];case"SpaceToBatchND":{const s=f("blockShape",e,t,n),a=f("paddings",e,t,n);return[r.spaceToBatchND(f("x",e,t,n),s,a)]}case"BatchToSpaceND":{const s=f("blockShape",e,t,n),a=f("crops",e,t,n);return[r.batchToSpaceND(f("x",e,t,n),s,a)]}case"DepthToSpace":{const s=f("blockSize",e,t,n),a=f("dataFormat",e,t,n).toUpperCase();return[r.depthToSpace(f("x",e,t,n),s,a)]}case"BroadcastTo":return[r.broadcastTo(f("x",e,t,n),f("shape",e,t,n))];case"BroadcastArgs":return[r.broadcastArgs(f("s0",e,t,n),f("s1",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $o(e,t,n,r,s=V){const a=((o,i,u)=>{switch(o.category){case"arithmetic":return s(()=>QS(o,i,u));case"basic_math":return s(()=>tT(o,i,u));case"control":return oT(o,i,u);case"convolution":return s(()=>iT(o,i,u));case"creation":return s(()=>uT(o,i,u));case"dynamic":return lT(o,i,u);case"evaluation":return s(()=>cT(o,i,u));case"image":return s(()=>dT(o,i,u));case"graph":return s(()=>hT(o,i,u));case"logical":return s(()=>mT(o,i,u));case"matrices":return s(()=>gT(o,i,u));case"normalization":return s(()=>yT(o,i,u));case"ragged":return s(()=>bT(o,i,u));case"reduction":return s(()=>wT(o,i,u));case"slice_join":return s(()=>NT(o,i,u));case"sparse":return s(()=>vT(o,i,u));case"spectral":return s(()=>ST(o,i,u));case"string":return s(()=>TT(o,i,u));case"transformation":return s(()=>ET(o,i,u));case"hash_table":return fT(o,i,u,r);case"custom":const l=pf(o.op);if(l&&l.customExecutor)return l.customExecutor(new JS(o,i,u));throw TypeError(`Custom op ${o.op} is not registered.`);default:throw TypeError(`Unknown op '${o.op}'. File an issue at https://github.com/tensorflow/tfjs/issues so we can add it, or register a custom execution with tf.registerOp()`)}})(e,t,n);return he(a)?a.then(o=>[].concat(o)):[].concat(a)}class _o{constructor(t={},n={},r={},s={},a){this.weightMap=t,this.tensorArrayMap=n,this.tensorListMap=r,this.functionMap=s,this.parseNodeNameCache=a,this.rootContext={id:0,frameName:"",iterationId:0},this.contexts=[this.rootContext],this.lastId=0,this.generateCurrentContextIds()}newFrame(t,n){return{id:t,frameName:n,iterationId:0}}set currentContext(t){this.contexts!==t&&(this.contexts=t,this.generateCurrentContextIds())}get currentContext(){return this.contexts}get currentContextId(){return this._currentContextIds[0]}get currentContextIds(){return this._currentContextIds}generateCurrentContextIds(){const t=[];for(let n=0;n<this.contexts.length-1;n++){const r=this.contexts.slice(0,this.contexts.length-n);t.push(this.contextIdforContexts(r))}t.push(""),this._currentContextIds=t}contextIdforContexts(t){return t?t.map(n=>n.id===0&&n.iterationId===0?"":`${n.frameName}-${n.iterationId}`).join("/"):""}enterFrame(t){this.contexts&&(this.lastId++,this.contexts=this.contexts.slice(),this.contexts.push(this.newFrame(this.lastId,t)),this._currentContextIds.unshift(this.contextIdforContexts(this.contexts)))}exitFrame(){if(this.contexts&&this.contexts.length>1)this.contexts=this.contexts.slice(),this.contexts.splice(-1),this.currentContextIds.shift();else throw new Error("Cannot exit frame, the context is empty")}nextIteration(){if(this.contexts&&this.contexts.length>0){this.contexts=this.contexts.slice(),this.lastId++;const t=Object.assign({},this.contexts[this.contexts.length-1]);t.iterationId+=1,t.id=this.lastId,this.contexts.splice(-1,1,t),this._currentContextIds.splice(0,1,this.contextIdforContexts(this.contexts))}else throw new Error("Cannot increase frame iteration, the context is empty")}getWeight(t){return this.weightMap[t]}addTensorArray(t){this.tensorArrayMap[t.id]=t}getTensorArray(t){return this.tensorArrayMap[t]}addTensorList(t){this.tensorListMap[t.id]=t}getTensorList(t){return this.tensorListMap[t]}dispose(t){for(const n in this.tensorArrayMap)this.tensorArrayMap[n].clearAndClose(t);for(const n in this.tensorListMap)this.tensorListMap[n].clearAndClose(t)}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ko(e,t,n,r){const s=new Set,a=[];let o=null,i=null;const u=new Set,l=new Set(Object.keys(e).map(p=>St(p)[0]));r=r||[];const h=new Set(r.map(p=>St(p.name)[0])),c=[...t];for(;c.length>0;){const p=c.pop();if((Ee(p)||DT(p)||FT(p))&&o==null&&(o=p,i=o.children.map(d=>d.name).filter(d=>s.has(d))),s.add(p.name),n[p.name]==null&&!l.has(p.name)&&!h.has(p.name)){if(p.inputs.length===0){a.push(p.name);continue}p.inputs.forEach(d=>{u.has(d.name)||(u.add(d.name),c.push(d))})}}return{inputs:e,outputs:t,usedNodes:s,missingInputs:a,dynamicNode:o,syncInputs:i}}function $T(e,t){const{usedNodes:n,inputs:r}=t,s=Object.keys(r).map(w=>St(w)[0]).map(w=>e.nodes[w]),a=e.initNodes||[],o=w=>n.has(typeof w=="string"?w:w.name);function i(w){return[...new Map(w.map(T=>[T.name,T])).values()]}const u=i([...s,...e.weights,...a]).filter(o),l=i([...u,...Object.values(e.nodes)]).filter(o),h=new Map(l.map(w=>[w.name,w])),c={};for(const w of l){c[w.name]=c[w.name]||0;for(const T of w.children)o(T)||(c[T.name]=Number.POSITIVE_INFINITY),c[T.name]=(c[T.name]||0)+1}const p=Object.entries(c).filter(([,w])=>w===0).map(([w])=>w),d=[...p];for(;p.length>0;){const w=p.pop(),T=h.get(w);for(const x of T.children.filter(o))--c[x.name]===0&&(d.push(x.name),p.push(x.name))}const g=d.map(w=>h.get(w)),N=_T(g,u);return kT(N,u),N}function _T(e,t){const n=new Map(e.map(o=>[o.name,o])),r=t.map(o=>o.name),s=new Set(r);for(;r.length>0;){const o=r.pop(),i=n.get(o);for(const u of i.children)!n.has(u.name)||s.has(u.name)||(s.add(u.name),r.push(u.name))}return e.filter(o=>s.has(o.name))}class Un extends Error{constructor(t){super(`NodesExecutionOrderError: ${t}`)}}function kT(e,t){const n=new Map(e.map((i,u)=>[i.name,u])),r=new Set(t.map(i=>i.name)),s=i=>r.has(typeof i=="string"?i:i.name),a=new Set(e.map(i=>i.name)),o=i=>a.has(typeof i=="string"?i:i.name);for(const i of e){for(const u of i.children.filter(o)){if(!n.has(u.name))throw new Un(`Child ${u.name} of node ${i.name} is unreachable.`);if(n.get(i.name)>n.get(u.name))throw new Un(`Node ${i.name} is scheduled to run after its child ${u.name}.`)}if(!s(i))for(const u of i.inputs){if(!n.has(u.name))throw new Un(`Input ${u.name} of node ${i.name} is unreachable.`);if(n.get(u.name)>n.get(i.name))throw new Un(`Node ${i.name} is scheduled to run before its input ${u.name}.`)}}}function IT(e){const t=new Map(e.map((i,u)=>[i.name,u])),n=Number.MAX_SAFE_INTEGER,r=e.map((i,u)=>Ee(i)?n:u),s=i=>{const u=r[t.get(i.name)];return u??-1},a=e.map((i,u)=>i.children.map(s).reduce((l,h)=>Math.max(l,h),r[u])),o=new Map;for(let i=0;i<e.length;++i){const u=a[i];if(u===n)continue;const l=e[i],h=e[u];o.has(h.name)||o.set(h.name,[]),o.get(h.name).push(l)}return o}const xT=new Set(["Switch","Merge","Enter","Exit","NextIteration","StatelessIf","StatelessWhile","if","While"]),AT=new Set(["NonMaxSuppressionV2","NonMaxSuppressionV3","NonMaxSuppressionV5","Where"]),OT=new Set(["HashTable","HashTableV2","LookupTableImport","LookupTableImportV2","LookupTableFind","LookupTableFindV2","LookupTableSize","LookupTableSizeV2"]);function Ee(e){return xT.has(e.op)}function DT(e){return AT.has(e.op)}function FT(e){return OT.has(e.op)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class wr{get weightIds(){return this.parent?this.parent.weightIds:this._weightIds}get functionExecutorMap(){return this.parent?this.parent.functionExecutorMap:this._functionExecutorMap}get weightMap(){return this.parent?this.parent.weightMap:this._weightMap}set weightMap(t){const n=Object.keys(t).map(r=>t[r].map(s=>s.id));this._weightIds=[].concat(...n),this._weightMap=t}set resourceManager(t){this._resourceManager=t}get inputs(){return this._inputs.map(t=>({name:t.name,shape:t.attrParams.shape?t.attrParams.shape.value:void 0,dtype:t.attrParams.dtype?t.attrParams.dtype.value:void 0}))}get outputs(){return this._outputs.map(t=>({name:t.name,shape:t.attrParams.shape?t.attrParams.shape.value:void 0,dtype:t.attrParams.dtype?t.attrParams.dtype.value:void 0}))}get inputNodes(){return this._inputs.map(t=>t.signatureKey||t.name)}get outputNodes(){return this._outputs.map(t=>{const n=t.signatureKey||t.name;return t.defaultOutput?`${n}:${t.defaultOutput}`:n})}get functions(){return Object.keys(this._functions).reduce((t,n)=>(t[n]=this._functions[n].signature,t),{})}constructor(t,n){this.graph=t,this.parent=n,this.compiledMap=new Map,this.parseNodeNameCache=new Map,this._weightMap={},this.SEPARATOR=",",this._functions={},this._functionExecutorMap={},this.keepIntermediateTensors=!1,this._outputs=t.outputs,this._inputs=t.inputs,this._initNodes=t.initNodes,this._signature=t.signature,this._functions=t.functions,t.functions!=null&&Object.keys(t.functions).forEach(r=>{this._functionExecutorMap[r]=new wr(t.functions[r],this)})}getCompilationKey(t,n){const r=t.map(a=>a.name).sort(),s=n.map(a=>a.name).sort();return r.join(this.SEPARATOR)+"--"+s.join(this.SEPARATOR)}compile(t,n){const r=ko(t,n,this.weightMap,this._initNodes),{missingInputs:s,dynamicNode:a,syncInputs:o}=r;if(a!=null)throw new Error(`This execution contains the node '${a.name}', which has the dynamic op '${a.op}'. Please use model.executeAsync() instead. Alternatively, to avoid the dynamic ops, specify the inputs [${o}]`);if(s.length>0){const l=n.map(c=>c.name),h=Object.keys(t);throw new Error(`Cannot compute the outputs [${l}] from the provided inputs [${h}]. Missing the following inputs: [${s}]`)}const i=$T(this.graph,r),u=IT(i);return{orderedNodes:i,nodeLiveUntilMap:u}}cloneAndKeepTensor(t){if(t==null)return null;const n=t.clone();return Rt(n),n}cloneTensorList(t){return t?t.map(r=>this.cloneAndKeepTensor(r)):null}cloneTensorMap(t){return Object.fromEntries(Object.entries(t).map(([n,r])=>[n,this.cloneTensorList(r)]))}execute(t,n){this.disposeIntermediateTensors(),t=this.mapInputs(t);const r=Object.keys(t).sort();this.checkInputs(t),this.checkInputShapeAndType(t),n=this.mapOutputs(n),this.checkOutputs(n);const s=r.map(p=>this.graph.nodes[St(p)[0]]),a=n.map(p=>St(p)[0]),o=new Set(a);let i=a.map(p=>this.graph.nodes[p]);i.length===0&&(i=this._outputs);const u=this.getCompilationKey(s,i);let l=this.compiledMap.get(u);l==null&&(l=this.compile(t,i),this.compiledMap.set(u,l));try{this.keepIntermediateTensors=M().getBool("KEEP_INTERMEDIATE_TENSORS")}catch(p){this.keepIntermediateTensors=!1,console.warn(p.message)}const h={},c={};return V(()=>{const p=new _o(this.weightMap,h,c,this.functionExecutorMap,this.parseNodeNameCache),d=Object.assign({},this.weightMap);this.keepIntermediateTensors&&(this.clonedTensorsMap=this.cloneTensorMap(this.weightMap)),Object.keys(t).forEach(T=>{const[x,$]=St(T,p),E=[];E[$]=t[T],d[x]=E,this.keepIntermediateTensors&&(this.clonedTensorsMap[x]=this.cloneTensorList(E))});const g=this.getFrozenTensorIds(d),{orderedNodes:N,nodeLiveUntilMap:w}=l;for(const T of N){if(d[T.name])continue;const x=$o(T,d,p,this._resourceManager);if(he(x))throw new Error(`The execution of the op '${T.op}' returned a promise. Please use model.executeAsync() instead.`);d[T.name]=x,this.keepIntermediateTensors&&(this.clonedTensorsMap[T.name]=this.cloneTensorList(x)),this.checkTensorForDisposalWithNodeLiveUntilInfo(T,d,p,g,o,w.get(T.name))}return this.parent==null&&p.dispose(g),n.map(T=>lt(T,d,p))})}getFrozenTensorIds(t){const n=[].concat.apply([],Object.keys(t).map(r=>t[r]).map(r=>r.map(s=>s.id)));return new Set(n)}checkTensorForDisposal(t,n,r,s,a,o,i){if(!(Ee(n)||o.has(t))){for(const u of r[t])u!=null&&(i[u.id]=(i[u.id]||0)+n.children.length);for(const u of n.inputs){if(Ee(u))continue;const l=No(u.name,r,s);if(l!=null)for(const h of l){if(!h||h.kept||a.has(h.id))continue;const c=i[h.id];c===1?(h.dispose(),delete i[h.id]):c!=null&&i[h.id]--}}}}checkTensorForDisposalWithNodeLiveUntilInfo(t,n,r,s,a,o){function i(u){return Ee(u)||a.has(u.name)}if(!(Ee(t)||o==null))for(const u of o){if(i(u))continue;const l=No(u.name,n,r);for(const h of l)!h||h.kept||s.has(h.id)||h.dispose()}}async executeAsync(t,n){return this._executeAsync(t,n)}disposeIntermediateTensors(){this.clonedTensorsMap&&(Object.values(this.clonedTensorsMap).forEach(t=>{for(const n of t)n&&!n.isDisposed&&n.dispose()}),this.clonedTensorsMap=null)}getIntermediateTensors(){return this.clonedTensorsMap}async _executeAsync(t,n,r=!1,s={},a={}){this.disposeIntermediateTensors(),r||(t=this.mapInputs(t),this.checkInputs(t),this.checkInputShapeAndType(t),n=this.mapOutputs(n),this.checkOutputs(n));try{this.keepIntermediateTensors=M().getBool("KEEP_INTERMEDIATE_TENSORS")}catch(p){this.keepIntermediateTensors=!1,console.warn(p.message)}const o=new _o(this.weightMap,s,a,this.functionExecutorMap,this.parseNodeNameCache);this.keepIntermediateTensors&&(this.clonedTensorsMap=this.cloneTensorMap(this.weightMap));const i=await this.executeWithControlFlow(t,o,n,r),u=n.map(p=>lt(p,i,o)),l=u.map(p=>p.id),h=Object.keys(t).map(p=>t[p].id),c=new Set([...l,...h,...this.weightIds]);return Object.values(i).forEach(p=>{p.forEach(d=>{d&&!d.isDisposed&&!c.has(d.id)&&d.dispose()})}),this.parent==null&&o.dispose(c),u}async executeFunctionAsync(t,n,r){const s=t.reduce((a,o,i)=>(a[this.inputs[i].name]=o,a),{});return this._executeAsync(s,this.outputNodes,!0,n,r)}async executeWithControlFlow(t,n,r,s){const a=Object.keys(t),o=a.map(E=>this.graph.nodes[St(E)[0]]),i=r.map(E=>St(E)[0]),u=new Set(i);let l=i.map(E=>this.graph.nodes[E]);l.length===0&&(l=this._outputs);const{usedNodes:h,missingInputs:c,dynamicNode:p,syncInputs:d}=ko(t,l,this.weightMap,this._initNodes),g=[...o,...this.graph.weights,...this._initNodes||[]].map(E=>({node:E,contexts:n.currentContext})),N=Object.assign({},this.weightMap);Object.keys(t).forEach(E=>{const[I,A]=St(E),F=[];F[A]=t[E],N[I]=F});const w={},T=this.getFrozenTensorIds(N),x={};for(;g.length>0;){const E=this.processStack(o,g,n,N,x,T,u,w,h);await Promise.all(E)}p==null&&!s&&console.warn("This model execution did not contain any nodes with control flow or dynamic output shapes. You can use model.execute() instead.");const $=l.filter(E=>!Ee(E)&&!lt(E.name,N,n)).map(E=>E.name);if($.length>0){let E="";throw p!=null&&(E=`Alternatively, to avoid the dynamic ops, use model.execute() and specify the inputs [${d}]`),new Error(`Cannot compute the outputs [${$}] from the provided inputs [${a}]. Consider providing the following inputs: [${c}]. ${E}`)}return N}processStack(t,n,r,s,a,o,i,u,l){const h=[];for(;n.length>0;){const c=n.pop();r.currentContext=c.contexts;let p="";if(c.node.op==="Enter"&&f("isConstant",c.node,s,r)&&([p]=Kt(c.node.name,r)),s[c.node.name]==null){const d=$o(c.node,s,r,this._resourceManager);p||([p]=Kt(c.node.name,r));const g=r.currentContext;he(d)?h.push(d.then(N=>(s[p]=N,this.keepIntermediateTensors&&(this.clonedTensorsMap[p]=this.cloneTensorList(N)),r.currentContext=g,this.checkTensorForDisposal(p,c.node,s,r,o,i,u),this.processChildNodes(c.node,n,r,s,a,l),N))):(s[p]=d,this.keepIntermediateTensors&&(this.clonedTensorsMap[p]=this.cloneTensorList(d)),this.checkTensorForDisposal(p,c.node,s,r,o,i,u),this.processChildNodes(c.node,n,r,s,a,l))}else this.processChildNodes(c.node,n,r,s,a,l)}return h}processChildNodes(t,n,r,s,a,o){t.children.forEach(i=>{const[u]=Kt(i.name,r);a[u]||!o.has(i.name)||(i.op==="Merge"?i.inputNames.some(l=>!!lt(l,s,r))&&(a[u]=!0,n.push({contexts:r.currentContext,node:i})):i.inputNames.every(l=>!!lt(l,s,r))&&(a[u]=!0,n.push({contexts:r.currentContext,node:i})))})}dispose(){Object.keys(this.weightMap).forEach(t=>this.weightMap[t].forEach(n=>n.dispose()))}checkInputShapeAndType(t){Object.keys(t).forEach(n=>{const r=t[n],[s]=St(n),a=this.graph.nodes[s];if(a.attrParams.shape&&a.attrParams.shape.value){const o=a.attrParams.shape.value,i=o.length===r.shape.length&&r.shape.every((u,l)=>o[l]===-1||o[l]===u);y(i,()=>`The shape of dict['${a.name}'] provided in model.execute(dict) must be [${o}], but was [${r.shape}]`)}a.attrParams.dtype&&a.attrParams.dtype.value&&y(r.dtype===a.attrParams.dtype.value,()=>`The dtype of dict['${a.name}'] provided in model.execute(dict) must be ${a.attrParams.dtype.value}, but was ${r.dtype}`)})}mapInputs(t){var n,r;const s={};for(const a in t){const o=(r=(n=this._signature)===null||n===void 0?void 0:n.inputs)===null||r===void 0?void 0:r[a];o!=null?s[o.name]=t[a]:s[a]=t[a]}return s}checkInputs(t){const n=Object.keys(t).filter(r=>{const[s]=St(r);return this.graph.nodes[s]==null});if(n.length>0)throw new Error(`The dict provided in model.execute(dict) has keys: [${n}] that are not part of graph`)}mapOutputs(t){return t.map(n=>{var r,s;const a=(s=(r=this._signature)===null||r===void 0?void 0:r.outputs)===null||s===void 0?void 0:s[n];return a!=null?a.name:n},{})}checkOutputs(t){t.forEach(n=>{const[r]=St(n);if(!this.graph.nodes[r])throw new Error(`The output '${n}' is not found in the graph`)})}}class RT{constructor(t={},n={}){this.hashTableNameToHandle=t,this.hashTableMap=n}addHashTable(t,n){this.hashTableNameToHandle[t]=n.handle,this.hashTableMap[n.id]=n}getHashTableHandleByName(t){return this.hashTableNameToHandle[t]}getHashTableById(t){return this.hashTableMap[t]}dispose(){for(const t in this.hashTableMap)this.hashTableMap[t].clearAndClose(),delete this.hashTableMap[t];for(const t in this.hashTableNameToHandle)this.hashTableNameToHandle[t].dispose(),delete this.hashTableNameToHandle[t]}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const BT="?tfjs-format=file",PT="model.json";class CT{get modelVersion(){return this.version}get inputNodes(){return this.executor.inputNodes}get outputNodes(){return this.executor.outputNodes}get inputs(){return this.executor.inputs}get outputs(){return this.executor.outputs}get weights(){return this.executor.weightMap}get metadata(){return this.artifacts.userDefinedMetadata}get modelSignature(){return this.signature}get modelStructuredOutputKeys(){return this.structuredOutputKeys}constructor(t,n={},r=Ua){this.modelUrl=t,this.loadOptions=n,this.version="n/a",this.io=r,n==null&&(this.loadOptions={}),this.resourceManager=new RT}findIOHandler(){const t=this.modelUrl;if(t.load!=null)this.handler=t;else if(this.loadOptions.requestInit!=null)this.handler=this.io.browserHTTPRequest(t,this.loadOptions);else{const n=this.io.getLoadHandlers(t,this.loadOptions);if(n.length===0)n.push(this.io.browserHTTPRequest(t,this.loadOptions));else if(n.length>1)throw new Error(`Found more than one (${n.length}) load handlers for URL '${[t]}'`);this.handler=n[0]}}load(){if(this.findIOHandler(),this.handler.load==null)throw new Error("Cannot proceed with model loading because the IOHandler provided does not have the `load` method implemented.");const t=this.handler.load();return he(t)?t.then(n=>n.getWeightStream==null?this.loadSync(n):this.loadStreaming(n)):this.loadSync(t)}loadSync(t){const n=this.io.decodeWeights(t.weightData,t.weightSpecs);return this.loadWithWeightMap(t,n)}async loadStreaming(t){if(t.getWeightStream==null)throw new Error("Model artifacts missing streamWeights function");const n=await uc(t.getWeightStream(),t.weightSpecs);return this.loadWithWeightMap(t,n)}loadWithWeightMap(t,n){this.artifacts=t;const r=this.artifacts.modelTopology;let s=this.artifacts.signature;if(this.artifacts.userDefinedMetadata!=null){const a=this.artifacts.userDefinedMetadata;a.signature!=null&&(s=a.signature),a.structuredOutputKeys!=null&&(this.structuredOutputKeys=a.structuredOutputKeys)}if(this.signature=s,this.version=`${r.versions.producer}.${r.versions.minConsumer}`,this.executor=new wr(vo.Instance.transformGraph(r,this.signature)),this.executor.weightMap=this.convertTensorMapToTensorsMap(n),this.executor.resourceManager=this.resourceManager,t.modelInitializer!=null&&t.modelInitializer.node!=null){const a=vo.Instance.transformGraph(t.modelInitializer);this.initializer=new wr(a),this.initializer.weightMap=this.executor.weightMap,this.initializer.resourceManager=this.resourceManager,this.initializerSignature=t.initializerSignature}return!0}async save(t,n){if(typeof t=="string"){const r=this.io.getSaveHandlers(t);if(r.length===0)throw new Error(`Cannot find any save handlers for URL '${t}'`);if(r.length>1)throw new Error(`Found more than one (${r.length}) save handlers for URL '${t}'`);t=r[0]}if(t.save==null)throw new Error("GraphModel.save() cannot proceed because the IOHandler provided does not have the `save` attribute defined.");return t.save(this.artifacts)}addStructuredOutputNames(t){if(this.structuredOutputKeys){const n=t instanceof et?[t]:t,r={};return n.forEach((s,a)=>r[this.structuredOutputKeys[a]]=s),r}return t}predict(t,n){const r=this.execute(t,this.outputNodes);return this.addStructuredOutputNames(r)}async predictAsync(t,n){const r=await this.executeAsync(t,this.outputNodes);return this.addStructuredOutputNames(r)}normalizeInputs(t){var n;if(!(t instanceof et)&&!Array.isArray(t)){const a=(n=this.signature)===null||n===void 0?void 0:n.inputs;if(a!=null)for(const o in a){const i=a[o];i.resourceId!=null&&(t[o]=this.resourceIdToCapturedInput[i.resourceId])}return t}t=Array.isArray(t)?t:[t];const r=Object.keys(this.resourceIdToCapturedInput).length;if(t.length+r!==this.inputNodes.length)throw new Error(`Input tensor count mismatch, the graph model has ${this.inputNodes.length-r} non-resource placeholders, while there are ${t.length} input tensors provided.`);let s=0;return this.inputNodes.reduce((a,o)=>{var i,u,l;const h=(l=(u=(i=this.signature)===null||i===void 0?void 0:i.inputs)===null||u===void 0?void 0:u[o])===null||l===void 0?void 0:l.resourceId;return h!=null?a[o]=this.resourceIdToCapturedInput[h]:a[o]=t[s++],a},{})}normalizeOutputs(t){return t=t||this.outputNodes,Array.isArray(t)?t:[t]}executeInitializerGraph(){return this.initializer==null?[]:this.initializerSignature==null?this.initializer.execute({},[]):this.initializer.execute({},Object.keys(this.initializerSignature.outputs))}async executeInitializerGraphAsync(){return this.initializer==null?[]:this.initializerSignature==null?this.initializer.executeAsync({},[]):this.initializer.executeAsync({},Object.keys(this.initializerSignature.outputs))}setResourceIdToCapturedInput(t){if(this.resourceIdToCapturedInput={},this.initializerSignature){const n=this.initializerSignature.outputs,r=Object.keys(n);for(let s=0;s<r.length;s++){const a=r[s],o=n[a];this.resourceIdToCapturedInput[o.resourceId]=t[s]}}}execute(t,n){this.resourceIdToCapturedInput==null&&this.setResourceIdToCapturedInput(this.executeInitializerGraph()),t=this.normalizeInputs(t),n=this.normalizeOutputs(n);const r=this.executor.execute(t,n);return r.length>1?r:r[0]}async executeAsync(t,n){this.resourceIdToCapturedInput==null&&this.setResourceIdToCapturedInput(await this.executeInitializerGraphAsync()),t=this.normalizeInputs(t),n=this.normalizeOutputs(n);const r=await this.executor.executeAsync(t,n);return r.length>1?r:r[0]}getIntermediateTensors(){return this.executor.getIntermediateTensors()}disposeIntermediateTensors(){this.executor.disposeIntermediateTensors()}convertTensorMapToTensorsMap(t){return Object.keys(t).reduce((n,r)=>(n[r]=[t[r]],n),{})}dispose(){this.executor.dispose(),this.initializer&&(this.initializer.dispose(),this.resourceIdToCapturedInput&&mt(this.resourceIdToCapturedInput)),this.resourceManager.dispose()}}async function mf(e,t={},n=Ua){if(e==null)throw new Error("modelUrl in loadGraphModel() cannot be null. Please provide a url or an IOHandler that loads the model");t==null&&(t={}),t.fromTFHub&&typeof e=="string"&&(e=LT(e));const r=new CT(e,t,n);return await r.load(),r}function LT(e){return e.endsWith("/")||(e=e+"/"),`${e}${PT}${BT}`}/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 *//*! *****************************************************************************
Copyright (c) Microsoft Corporation.

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
***************************************************************************** */var As=function(e,t){return As=Object.setPrototypeOf||{__proto__:[]}instanceof Array&&function(n,r){n.__proto__=r}||function(n,r){for(var s in r)r.hasOwnProperty(s)&&(n[s]=r[s])},As(e,t)};function gf(e,t){As(e,t);function n(){this.constructor=e}e.prototype=t===null?Object.create(t):(n.prototype=t.prototype,new n)}var Lt=function(){return Lt=Object.assign||function(t){for(var n,r=1,s=arguments.length;r<s;r++){n=arguments[r];for(var a in n)Object.prototype.hasOwnProperty.call(n,a)&&(t[a]=n[a])}return t},Lt.apply(this,arguments)};function Vt(e,t,n,r){function s(a){return a instanceof n?a:new n(function(o){o(a)})}return new(n||(n=Promise))(function(a,o){function i(h){try{l(r.next(h))}catch(c){o(c)}}function u(h){try{l(r.throw(h))}catch(c){o(c)}}function l(h){h.done?a(h.value):s(h.value).then(i,u)}l((r=r.apply(e,[])).next())})}function Wt(e,t){var n={label:0,sent:function(){if(a[0]&1)throw a[1];return a[1]},trys:[],ops:[]},r,s,a,o;return o={next:i(0),throw:i(1),return:i(2)},typeof Symbol=="function"&&(o[Symbol.iterator]=function(){return this}),o;function i(l){return function(h){return u([l,h])}}function u(l){if(r)throw new TypeError("Generator is already executing.");for(;n;)try{if(r=1,s&&(a=l[0]&2?s.return:l[0]?s.throw||((a=s.return)&&a.call(s),0):s.next)&&!(a=a.call(s,l[1])).done)return a;switch(s=0,a&&(l=[l[0]&2,a.value]),l[0]){case 0:case 1:a=l;break;case 4:return n.label++,{value:l[1],done:!1};case 5:n.label++,s=l[1],l=[0];continue;case 7:l=n.ops.pop(),n.trys.pop();continue;default:if(a=n.trys,!(a=a.length>0&&a[a.length-1])&&(l[0]===6||l[0]===2)){n=0;continue}if(l[0]===3&&(!a||l[1]>a[0]&&l[1]<a[3])){n.label=l[1];break}if(l[0]===6&&n.label<a[1]){n.label=a[1],a=l;break}if(a&&n.label<a[2]){n.label=a[2],n.ops.push(l);break}a[2]&&n.ops.pop(),n.trys.pop();continue}l=t.call(e,n)}catch(h){l=[6,h],s=0}finally{r=a=0}if(l[0]&5)throw l[1];return{value:l[0]?l[1]:void 0,done:!0}}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function yf(e){var t=e.shape[2],n=Ys(e,2),r=O(n,[-1]);return Tn(r,t)}function zT(e,t){return B(e,t)}function qn(e,t){return V(function(){return j(Qe(e,U(t)),"int32")})}function MT(e,t){var n=t.shape,r=n[0],s=n[1],a=n[2];return V(function(){var o=yf(t),i=_t(me(0,a,1,"int32"),1),u=j(H(o,i),"int32"),l=O(u,[r,s]),h=z(l,U(1,"int32"));return W(zT(h,e),U(1,"int32"))})}function VT(e){var t=e.shape,n=t[0],r=t[1],s=t[2];return V(function(){var a=yf(e),o=_t(me(0,s,1,"int32"),1),i=j(H(a,o),"int32");return O(i,[n,r])})}/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var bf=function(){function e(t,n){this.model=t,this.outputStride=n;var r=this.model.inputs[0].shape;y(r[1]===-1&&r[2]===-1,function(){return"Input shape [".concat(r[1],", ").concat(r[2],"] ")+"must both be equal to or -1"})}return e.prototype.predict=function(t){var n=this;return V(function(){var r=n.preprocessInput(j(t,"float32")),s=_t(r,0),a=n.model.predict(s),o=a.map(function(u){return Mt(u,[0])}),i=n.nameOutputResults(o);return{heatmapScores:Qt(i.heatmap),offsets:i.offsets,displacementFwd:i.displacementFwd,displacementBwd:i.displacementBwd,segmentation:i.segmentation,partHeatmaps:i.partHeatmaps,longOffsets:i.longOffsets,partOffsets:i.partOffsets}})},e.prototype.dispose=function(){this.model.dispose()},e}();/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var WT=function(e){gf(t,e);function t(){return e!==null&&e.apply(this,arguments)||this}return t.prototype.preprocessInput=function(n){return V(function(){return W(Y(n,127.5),1)})},t.prototype.nameOutputResults=function(n){var r=n[0],s=n[1],a=n[2],o=n[3],i=n[4],u=n[5],l=n[6],h=n[7];return{offsets:r,segmentation:s,partHeatmaps:a,longOffsets:o,heatmap:i,displacementFwd:u,displacementBwd:l,partOffsets:h}},t}(bf);/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var Lr=["nose","leftEye","rightEye","leftEar","rightEar","leftShoulder","rightShoulder","leftElbow","rightElbow","leftWrist","rightWrist","leftHip","rightHip","leftKnee","rightKnee","leftAnkle","rightAnkle"],Ft=Lr.length,Nr=Lr.reduce(function(e,t,n){return e[t]=n,e},{}),UT=[["leftHip","leftShoulder"],["leftElbow","leftShoulder"],["leftElbow","leftWrist"],["leftHip","leftKnee"],["leftKnee","leftAnkle"],["rightHip","rightShoulder"],["rightElbow","rightShoulder"],["rightElbow","rightWrist"],["rightHip","rightKnee"],["rightKnee","rightAnkle"],["leftShoulder","rightShoulder"],["leftHip","rightHip"]],qT=[["nose","leftEye"],["leftEye","leftEar"],["nose","rightEye"],["rightEye","rightEar"],["nose","leftShoulder"],["leftShoulder","leftElbow"],["leftElbow","leftWrist"],["leftShoulder","leftHip"],["leftHip","leftKnee"],["leftKnee","leftAnkle"],["nose","rightShoulder"],["rightShoulder","rightElbow"],["rightElbow","rightWrist"],["rightShoulder","rightHip"],["rightHip","rightKnee"],["rightKnee","rightAnkle"]];UT.map(function(e){var t=e[0],n=e[1];return[Nr[t],Nr[n]]});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ja(e,t,n){var r=e[0],s=e[1],a=t[0],o=t[1],i=n.top,u=n.bottom,l=n.left,h=n.right,c=a/(i+u+r),p=o/(l+h+s);return[p,c]}function wf(e,t,n,r){return{y:r.get(e,t,n),x:r.get(e,t,n+Ft)}}function Nf(e,t,n){var r=e.heatmapY,s=e.heatmapX,a=e.id,o=wf(r,s,a,n),i=o.y,u=o.x;return{x:e.heatmapX*t+u,y:e.heatmapY*t+i}}function Io(e,t,n){return e<t?t:e>n?n:e}function HT(e,t,n,r){var s=n-e,a=r-t;return s*s+a*a}function xo(e,t){return{x:e.x+t.x,y:e.y+t.y}}/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function jT(e,t,n){n===void 0&&(n=.3);for(var r=0,s=0,a=0;a<e.length;a++)t.keypoints[a].score>n&&(s+=1,r+=Math.pow(e[a].x-t.keypoints[a].position.x,2)+Math.pow(e[a].y-t.keypoints[a].position.y,2));return s===0?r=1/0:r=r/s,r}function GT(e,t,n,r){var s=t[0],a=t[1],o=n[0],i=n[1],u=Math.round(((s+e.y+1)*i-1)/r),l=Math.round(((a+e.x+1)*o-1)/r);return{x:l,y:u}}function KT(e,t,n,r,s,a,o){for(var i=o[0],u=o[1],l=n(e),h=l.y*r+l.x,c=s[Ft*(2*h)+t],p=s[Ft*(2*h+1)+t],d=e.y+c,g=e.x+p,N=0;N<a;N++){d=Math.min(d,i-1),g=Math.min(g,u-1);var w=n({x:g,y:d}),T=w.y*r+w.x;c=s[Ft*(2*T)+t],p=s[Ft*(2*T+1)+t],d=d+c,g=g+p}return{x:g,y:d}}function vf(e,t,n,r,s,a,o,i,u,l){for(var h=s[0],c=s[1],p=a[0],d=a[1],g=i[0],N=i[1],w=[],T=function(R){return GT(R,[h,c],[p,d],u)},x=0;x<r;x++){var $=KT(e,x,T,o,t,l,[g,N]);w.push($)}for(var E=-1,I=1/0,A=0;A<n.length;A++){var F=jT(w,n[A]);F<I&&(E=A,I=F)}return E}function Sf(e,t){var n=e[0],r=e[1],s=Math.round((r-1)/t+1),a=Math.round((n-1)/t+1);return[s,a]}function XT(e,t,n,r,s,a,o,i,u,l){var h=o[0],c=o[1];l===void 0&&(l=5);for(var p=n.map(function(R){return new Uint8Array(r*s).fill(0)}),d=i.top,g=i.left,N=ja([r,s],[h,c],i),w=N[0],T=N[1],x=Sf([h,c],a)[0],$=0;$<r;$+=1)for(var E=0;E<s;E+=1){var I=$*s+E,A=e[I];if(A===1){var F=vf({x:E,y:$},t,n,l,[d,g],[w,T],x,[r,s],a,u);F>=0&&(p[F][I]=1)}}return p}function YT(e,t,n,r,s,a,o,i,u,l,h){var c=i[0],p=i[1];h===void 0&&(h=5);for(var d=r.map(function(k){return new Int32Array(s*a).fill(-1)}),g=u.top,N=u.left,w=ja([s,a],[c,p],u),T=w[0],x=w[1],$=Sf([c,p],o)[0],E=0;E<s;E+=1)for(var I=0;I<a;I+=1){var A=E*a+I,F=e[A];if(F===1){var R=vf({x:I,y:E},t,r,h,[g,N],[T,x],$,[s,a],o,l);R>=0&&(d[R][A]=n[A])}}return d}/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Tf(e,t,n,r,s,a,o,i,u,l,h){for(var c=o[0],p=o[1],d=e.shape,g=d[0],N=d[1],w=t.shape.slice(0,2),T=w[0],x=w[1],$=O(t,[T,x,2,Ft]),E=new Float32Array(h*Ft*3).fill(0),I=0;I<n.length;I++)for(var A=I*Ft*3,F=n[I],R=0;R<Ft;R++){var k=F.keypoints[R],_=A+R*3;E[_]=k.score,E[_+1]=k.position.y,E[_+2]=k.position.x}var b=ja([r,s],[c,p],i),D=b[0],P=b[1],C=xt(E,[h,Ft,3]),L=i.top,q=i.left,G={variableNames:["segmentation","longOffsets","poses"],outputShape:[g,N],userCode:`
    int convertToPositionInOutput(int pos, int pad, float scale, int stride) {
      return round(((float(pos + pad) + 1.0) * scale - 1.0) / float(stride));
    }

    float convertToPositionInOutputFloat(
        int pos, int pad, float scale, int stride) {
      return ((float(pos + pad) + 1.0) * scale - 1.0) / float(stride);
    }

    float dist(float x1, float y1, float x2, float y2) {
      return pow(x1 - x2, 2.0) + pow(y1 - y2, 2.0);
    }

    float sampleLongOffsets(float h, float w, int d, int k) {
      float fh = fract(h);
      float fw = fract(w);
      int clH = int(ceil(h));
      int clW = int(ceil(w));
      int flH = int(floor(h));
      int flW = int(floor(w));
      float o11 = getLongOffsets(flH, flW, d, k);
      float o12 = getLongOffsets(flH, clW, d, k);
      float o21 = getLongOffsets(clH, flW, d, k);
      float o22 = getLongOffsets(clH, clW, d, k);
      float o1 = mix(o11, o12, fw);
      float o2 = mix(o21, o22, fw);
      return mix(o1, o2, fh);
    }

    int findNearestPose(int h, int w) {
      float prob = getSegmentation(h, w);
      if (prob < 1.0) {
        return -1;
      }

      // Done(Tyler): convert from output space h/w to strided space.
      float stridedH = convertToPositionInOutputFloat(
        h, `.concat(L,", ").concat(P,", ").concat(a,`);
      float stridedW = convertToPositionInOutputFloat(
        w, `).concat(q,", ").concat(D,", ").concat(a,`);

      float minDist = 1000000.0;
      int iMin = -1;
      for (int i = 0; i < `).concat(h,`; i++) {
        float curDistSum = 0.0;
        int numKpt = 0;
        for (int k = 0; k < `).concat(Ft,`; k++) {
          float dy = sampleLongOffsets(stridedH, stridedW, 0, k);
          float dx = sampleLongOffsets(stridedH, stridedW, 1, k);

          float y = float(h) + dy;
          float x = float(w) + dx;

          for (int s = 0; s < `).concat(u,`; s++) {
            int yRounded = round(min(y, float(`).concat(r-1,`)));
            int xRounded = round(min(x, float(`).concat(s-1,`)));

            float yStrided = convertToPositionInOutputFloat(
              yRounded, `).concat(L,", ").concat(P,", ").concat(a,`);
            float xStrided = convertToPositionInOutputFloat(
              xRounded, `).concat(q,", ").concat(D,", ").concat(a,`);

            float dy = sampleLongOffsets(yStrided, xStrided, 0, k);
            float dx = sampleLongOffsets(yStrided, xStrided, 1, k);

            y = y + dy;
            x = x + dx;
          }

          float poseScore = getPoses(i, k, 0);
          float poseY = getPoses(i, k, 1);
          float poseX = getPoses(i, k, 2);
          if (poseScore > `).concat(l,`) {
            numKpt = numKpt + 1;
            curDistSum = curDistSum + dist(x, y, poseX, poseY);
          }
        }
        if (numKpt > 0 && curDistSum / float(numKpt) < minDist) {
          minDist = curDistSum / float(numKpt);
          iMin = i;
        }
      }
      return iMin;
    }

    void main() {
        ivec2 coords = getOutputCoords();
        int nearestPose = findNearestPose(coords[0], coords[1]);
        setOutput(float(nearestPose));
      }
  `)},tt=Hs();return tt.compileAndRun(G,[e,$,C])}/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ZT(e,t){return V(function(){return j(Rn(e,U(t)),"int32")})}function JT(e,t,n){return V(function(){return W(B(j(Rn(e,U(n)),"int32"),z(t,1)),1)})}function Ef(){return qs()==="webgl"}function QT(e,t,n,r,s,a,o,i,u,l,h,c){var p=o[0],d=o[1];return u===void 0&&(u=.2),l===void 0&&(l=8),h===void 0&&(h=.3),c===void 0&&(c=10),Vt(this,void 0,void 0,function(){var g,N,w,T,x;return Wt(this,function($){switch($.label){case 0:return g=n.filter(function(E){return E.score>=u}),Ef()?(w=V(function(){var E=Tf(e,t,g,r,s,a,[p,d],i,l,h,c),I=Us().makeTensorFromDataId(E.dataId,E.shape,E.dtype);return g.map(function(A,F){return ZT(I,F)})}),[4,Promise.all(w.map(function(E){return E.data()}))]):[3,2];case 1:return N=$.sent(),w.forEach(function(E){return E.dispose()}),[3,5];case 2:return[4,e.data()];case 3:return T=$.sent(),[4,t.data()];case 4:x=$.sent(),N=XT(T,x,g,r,s,a,[p,d],i,l),$.label=5;case 5:return[2,N.map(function(E,I){return{data:E,pose:g[I],width:s,height:r}})]}})})}function tE(e,t,n,r,s,a,o,i,u,l,h,c,p){var d=i[0],g=i[1];return l===void 0&&(l=.2),h===void 0&&(h=8),c===void 0&&(c=.3),p===void 0&&(p=10),Vt(this,void 0,void 0,function(){var N,w,T,x,$,E;return Wt(this,function(I){switch(I.label){case 0:return N=r.filter(function(A){return A.score>=l}),Ef()?(T=V(function(){var A=Tf(e,t,N,s,a,o,[d,g],u,h,c,p),F=Us().makeTensorFromDataId(A.dataId,A.shape,A.dtype);return N.map(function(R,k){return JT(F,n,k)})}),[4,Promise.all(T.map(function(A){return A.data()}))]):[3,2];case 1:return w=I.sent(),T.forEach(function(A){return A.dispose()}),[3,6];case 2:return[4,e.data()];case 3:return x=I.sent(),[4,t.data()];case 4:return $=I.sent(),[4,n.data()];case 5:E=I.sent(),w=YT(x,$,E,N,s,a,o,[d,g],u,h),I.label=6;case 6:return[2,w.map(function(A,F){return{pose:N[F],data:A,height:s,width:a}})]}})})}/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Kr(e){return Math.floor(e/2)}var eE=function(){function e(t,n){this.priorityQueue=new Array(t),this.numberOfElements=-1,this.getElementValue=n}return e.prototype.enqueue=function(t){this.priorityQueue[++this.numberOfElements]=t,this.swim(this.numberOfElements)},e.prototype.dequeue=function(){var t=this.priorityQueue[0];return this.exchange(0,this.numberOfElements--),this.sink(0),this.priorityQueue[this.numberOfElements+1]=null,t},e.prototype.empty=function(){return this.numberOfElements===-1},e.prototype.size=function(){return this.numberOfElements+1},e.prototype.all=function(){return this.priorityQueue.slice(0,this.numberOfElements+1)},e.prototype.max=function(){return this.priorityQueue[0]},e.prototype.swim=function(t){for(;t>0&&this.less(Kr(t),t);)this.exchange(t,Kr(t)),t=Kr(t)},e.prototype.sink=function(t){for(;2*t<=this.numberOfElements;){var n=2*t;if(n<this.numberOfElements&&this.less(n,n+1)&&n++,!this.less(t,n))break;this.exchange(t,n),t=n}},e.prototype.getValueAt=function(t){return this.getElementValue(this.priorityQueue[t])},e.prototype.less=function(t,n){return this.getValueAt(t)<this.getValueAt(n)},e.prototype.exchange=function(t,n){var r=this.priorityQueue[t];this.priorityQueue[t]=this.priorityQueue[n],this.priorityQueue[n]=r},e}();/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function nE(e,t,n,r,s,a){for(var o=a.shape,i=o[0],u=o[1],l=!0,h=Math.max(n-s,0),c=Math.min(n+s+1,i),p=h;p<c;++p){for(var d=Math.max(r-s,0),g=Math.min(r+s+1,u),N=d;N<g;++N)if(a.get(p,N,e)>t){l=!1;break}if(!l)break}return l}function rE(e,t,n){for(var r=n.shape,s=r[0],a=r[1],o=r[2],i=new eE(s*a*o,function(p){var d=p.score;return d}),u=0;u<s;++u)for(var l=0;l<a;++l)for(var h=0;h<o;++h){var c=n.get(u,l,h);c<e||nE(h,c,u,l,t,n)&&i.enqueue({score:c,part:{heatmapY:u,heatmapX:l,id:h}})}return i}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var $f=qT.map(function(e){var t=e[0],n=e[1];return[Nr[t],Nr[n]]}),Xr=$f.map(function(e){var t=e[1];return t}),Ao=$f.map(function(e){var t=e[0];return t});function sE(e,t,n){var r=n.shape[2]/2;return{y:n.get(t.y,t.x,e),x:n.get(t.y,t.x,r+e)}}function Yr(e,t,n,r){return{y:Io(Math.round(e.y/t),0,n-1),x:Io(Math.round(e.x/t),0,r-1)}}function Oo(e,t,n,r,s,a,o,i){i===void 0&&(i=2);for(var u=r.shape,l=u[0],h=u[1],c=Yr(t.position,a,l,h),p=sE(e,c,o),d=xo(t.position,p),g=d,N=0;N<i;N++){var w=Yr(g,a,l,h),T=wf(w.y,w.x,n,s);g=xo({x:w.x*a,y:w.y*a},{x:T.x,y:T.y})}var x=Yr(g,a,l,h),$=r.get(x.y,x.x,n);return{position:g,part:Lr[n],score:$}}function aE(e,t,n,r,s,a){var o=t.shape[2],i=Xr.length,u=new Array(o),l=e.part,h=e.score,c=Nf(l,r,n);u[l.id]={score:h,part:Lr[l.id],position:c};for(var p=i-1;p>=0;--p){var d=Xr[p],g=Ao[p];u[d]&&!u[g]&&(u[g]=Oo(p,u[d],g,t,n,r,a))}for(var p=0;p<i;++p){var d=Ao[p],g=Xr[p];u[d]&&!u[g]&&(u[g]=Oo(p,u[d],g,t,n,r,s))}return u}/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _f(e,t,n,r){var s=n.x,a=n.y;return e.some(function(o){var i=o.keypoints,u=i[r].position;return HT(a,s,u.y,u.x)<=t})}function oE(e,t,n){var r=n.reduce(function(s,a,o){var i=a.position,u=a.score;return _f(e,t,i,o)||(s+=u),s},0);return r/=n.length}var iE=1;function Hn(e,t,n,r,s,a,o,i){o===void 0&&(o=.5),i===void 0&&(i=20);for(var u=[],l=rE(o,iE,e),h=i*i;u.length<a&&!l.empty();){var c=l.dequeue(),p=Nf(c.part,s,t);if(!_f(u,h,p,c.part.id)){var d=aE(c,e,t,s,n,r),g=oE(u,h,d);u.push({keypoints:d,score:g})}}return u}/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var uE=[-123.15,-115.9,-103.06],lE=function(e){gf(t,e);function t(){return e!==null&&e.apply(this,arguments)||this}return t.prototype.preprocessInput=function(n){return z(n,uE)},t.prototype.nameOutputResults=function(n){var r=n[0],s=n[1],a=n[2],o=n[3],i=n[4],u=n[5],l=n[6],h=n[7];return{offsets:i,segmentation:l,partHeatmaps:u,longOffsets:o,heatmap:a,displacementFwd:s,displacementBwd:r,partOffsets:h}},t}(bf);/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var Do="https://storage.googleapis.com/tfjs-models/savedmodel/bodypix/resnet50/",Fo="https://storage.googleapis.com/tfjs-models/savedmodel/bodypix/mobilenet/";function cE(e,t){var n="model-stride".concat(e,".json");return t===4?Do+"float/"+n:Do+"quant".concat(t,"/")+n}function hE(e,t,n){var r={1:"100",.75:"075",.5:"050"},s="model-stride".concat(e,".json");return n===4?Fo+"float/".concat(r[t],"/")+s:Fo+"quant".concat(n,"/").concat(r[t],"/")+s}/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * =============================================================================
 */var Pe;function pE(e){if("offsetHeight"in e&&e.offsetHeight!==0&&"offsetWidth"in e&&e.offsetWidth!==0)return[e.offsetHeight,e.offsetWidth];if(e.height!=null&&e.width!=null)return[e.height,e.width];throw new Error("HTMLImageElement must have height and width attributes set.")}function fE(e){return e.hasAttribute("height")&&e.hasAttribute("width")?[e.height,e.width]:[e.videoHeight,e.videoWidth]}function on(e){if(typeof HTMLCanvasElement<"u"&&e instanceof HTMLCanvasElement||typeof OffscreenCanvas<"u"&&e instanceof OffscreenCanvas||typeof HTMLImageElement<"u"&&e instanceof HTMLImageElement)return pE(e);if(typeof ImageData<"u"&&e instanceof ImageData)return[e.height,e.width];if(typeof HTMLVideoElement<"u"&&e instanceof HTMLVideoElement)return fE(e);if(e instanceof et)return[e.shape[0],e.shape[1]];throw new Error("error: Unknown input type: ".concat(e,"."))}function dE(e,t){return(e-1)%t===0}function Ro(e,t){return dE(e,t)?e:Math.floor(e/t)*t+1}var un={low:"low",medium:"medium",high:"high",full:"full"},mE=(Pe={},Pe[un.low]=.25,Pe[un.medium]=.5,Pe[un.high]=.75,Pe[un.full]=1,Pe),Bo=.1,Po=2;function gE(e){if(typeof e=="string"){var t=mE[e];return y(typeof t=="number",function(){return"string value of inputResolution must be one of ".concat(Object.values(un).join(",")," but was ").concat(e,".")}),t}else return y(typeof e=="number"&&e<=Po&&e>=Bo,function(){return"inputResolution must be a string or number between ".concat(Bo," and ").concat(Po,", but ")+"was ".concat(e)}),e}function jn(e,t,n){var r=n[0],s=n[1],a=gE(e);return[Ro(r*a,t),Ro(s*a,t)]}function yE(e){return e instanceof et?e:Jp(e)}function Ce(e,t,n,r,s){var a=t[0],o=t[1],i=n[0],u=n[1],l=r[0],h=l[0],c=l[1],p=r[1],d=p[0],g=p[1];return V(function(){var N=zn.resizeBilinear(e,[i,u],!0);return N=Qt(N),bE(N,[a,o],[[h,c],[d,g]])})}function bE(e,t,n){var r=t[0],s=t[1],a=n[0],o=a[0],i=a[1],u=n[1],l=u[0],h=u[1];return V(function(){var c=_t(e);return Mt(zn.cropAndResize(c,[[o/(r+o+i-1),l/(s+l+h-1),(o+r-1)/(r+o+i-1),(l+s-1)/(s+l+h-1)]],[0],[r,s]),[0])})}function Gn(e,t){var n=t[0],r=t[1],s=on(e),a=s[0],o=s[1],i=r/n,u=o/a,l=[0,0,0,0],h=l[0],c=l[1],p=l[2],d=l[3];u<i?(h=0,c=0,p=Math.round(.5*(i*a-o)),d=Math.round(.5*(i*a-o))):(h=Math.round(.5*(1/i*o-a)),c=Math.round(.5*(1/i*o-a)),p=0,d=0);var g=V(function(){var N=yE(e);return N=ya(N,[[h,c],[p,d],[0,0]]),zn.resizeBilinear(N,[n,r])});return{resized:g,padding:{top:h,left:p,right:d,bottom:c}}}function Kn(e){return Vt(this,void 0,void 0,function(){return Wt(this,function(t){return[2,Promise.all(e.map(function(n){return n.buffer()}))]})})}function wE(e,t,n,r,s){return r===void 0&&(r=0),s===void 0&&(s=0),{score:e.score,keypoints:e.keypoints.map(function(a){var o=a.score,i=a.part,u=a.position;return{score:o,part:i,position:{x:u.x*n+s,y:u.y*t+r}}})}}function NE(e,t,n,r,s){return r===void 0&&(r=0),s===void 0&&(s=0),n===1&&t===1&&r===0&&s===0?e:e.map(function(a){return wE(a,t,n,r,s)})}function Xn(e,t,n,r,s){var a=t[0],o=t[1],i=n[0],u=n[1],l=(a+r.top+r.bottom)/i,h=(o+r.left+r.right)/u,c=NE(e,l,h,-r.top,-r.left);return c}var kf={architecture:"MobileNetV1",outputStride:16,quantBytes:4,multiplier:.75},Co=["MobileNetV1","ResNet50"],Lo={MobileNetV1:[8,16,32],ResNet50:[32,16]},zo={MobileNetV1:[.5,.75,1],ResNet50:[1]},Mo=[1,2,4];function vE(e){if(e=e||kf,e.architecture==null&&(e.architecture="MobileNetV1"),Co.indexOf(e.architecture)<0)throw new Error("Invalid architecture ".concat(e.architecture,". ")+"Should be one of ".concat(Co));if(e.outputStride==null&&(e.outputStride=16),Lo[e.architecture].indexOf(e.outputStride)<0)throw new Error("Invalid outputStride ".concat(e.outputStride,". ")+"Should be one of ".concat(Lo[e.architecture]," ")+"for architecture ".concat(e.architecture,"."));if(e.multiplier==null&&(e.multiplier=1),zo[e.architecture].indexOf(e.multiplier)<0)throw new Error("Invalid multiplier ".concat(e.multiplier,". ")+"Should be one of ".concat(zo[e.architecture]," ")+"for architecture ".concat(e.architecture,"."));if(e.quantBytes==null&&(e.quantBytes=4),Mo.indexOf(e.quantBytes)<0)throw new Error("Invalid quantBytes ".concat(e.quantBytes,". ")+"Should be one of ".concat(Mo," ")+"for architecture ".concat(e.architecture,"."));return e}var Yn={flipHorizontal:!1,internalResolution:"medium",segmentationThreshold:.7,maxDetections:10,scoreThreshold:.4,nmsRadius:20},Zn={flipHorizontal:!1,internalResolution:"medium",segmentationThreshold:.7,maxDetections:10,scoreThreshold:.4,nmsRadius:20,minKeypointScore:.3,refineSteps:10};function Vo(e){var t=e.segmentationThreshold,n=e.maxDetections,r=e.scoreThreshold,s=e.nmsRadius;if(t<0||t>1)throw new Error("segmentationThreshold ".concat(t,". ")+"Should be in range [0.0, 1.0]");if(n<=0)throw new Error("Invalid maxDetections ".concat(n,". ")+"Should be > 0");if(r<0||r>1)throw new Error("Invalid scoreThreshold ".concat(r,". ")+"Should be in range [0.0, 1.0]");if(s<=0)throw new Error("Invalid nmsRadius ".concat(s,"."))}function Wo(e){var t=e.segmentationThreshold,n=e.maxDetections,r=e.scoreThreshold,s=e.nmsRadius,a=e.minKeypointScore,o=e.refineSteps;if(t<0||t>1)throw new Error("segmentationThreshold ".concat(t,". ")+"Should be in range [0.0, 1.0]");if(n<=0)throw new Error("Invalid maxDetections ".concat(n,". ")+"Should be > 0");if(r<0||r>1)throw new Error("Invalid scoreThreshold ".concat(r,". ")+"Should be in range [0.0, 1.0]");if(s<=0)throw new Error("Invalid nmsRadius ".concat(s,"."));if(a<0||a>1)throw new Error("Invalid minKeypointScore ".concat(a,".")+"Should be in range [0.0, 1.0]");if(o<=0||o>20)throw new Error("Invalid refineSteps ".concat(o,".")+"Should be in range [1, 20]")}var If=function(){function e(t){this.baseModel=t}return e.prototype.predictForPersonSegmentation=function(t){var n=this.baseModel.predict(t),r=n.segmentation,s=n.heatmapScores,a=n.offsets,o=n.displacementFwd,i=n.displacementBwd;return{segmentLogits:r,heatmapScores:s,offsets:a,displacementFwd:o,displacementBwd:i}},e.prototype.predictForPersonSegmentationAndPart=function(t){var n=this.baseModel.predict(t),r=n.segmentation,s=n.partHeatmaps,a=n.heatmapScores,o=n.offsets,i=n.displacementFwd,u=n.displacementBwd;return{segmentLogits:r,partHeatmapLogits:s,heatmapScores:a,offsets:o,displacementFwd:i,displacementBwd:u}},e.prototype.predictForMultiPersonInstanceSegmentationAndPart=function(t){var n=this.baseModel.predict(t),r=n.segmentation,s=n.longOffsets,a=n.heatmapScores,o=n.offsets,i=n.displacementFwd,u=n.displacementBwd,l=n.partHeatmaps;return{segmentLogits:r,longOffsets:s,heatmapScores:a,offsets:o,displacementFwd:i,displacementBwd:u,partHeatmaps:l}},e.prototype.segmentPersonActivation=function(t,n,r){var s=this;r===void 0&&(r=.5);var a=on(t),o=a[0],i=a[1],u=jn(n,this.baseModel.outputStride,[o,i]),l=Gn(t,u),h=l.resized,c=l.padding,p=V(function(){var x=s.predictForPersonSegmentation(h),$=x.segmentLogits,E=x.heatmapScores,I=x.offsets,A=x.displacementFwd,F=x.displacementBwd,R=h.shape,k=R[0],_=R[1],b=Ce($,[o,i],[k,_],[[c.top,c.bottom],[c.left,c.right]]);return{segmentation:qn(Mt(b),r),heatmapScores:E,offsets:I,displacementFwd:A,displacementBwd:F}}),d=p.segmentation,g=p.heatmapScores,N=p.offsets,w=p.displacementFwd,T=p.displacementBwd;return h.dispose(),{segmentation:d,heatmapScores:g,offsets:N,displacementFwd:w,displacementBwd:T,padding:c,internalResolutionHeightAndWidth:u}},e.prototype.segmentPerson=function(t,n){return n===void 0&&(n=Yn),Vt(this,void 0,void 0,function(){var r,s,a,o,i,u,l,h,c,p,d,g,N,w,T,x,$,E;return Wt(this,function(I){switch(I.label){case 0:return n=Lt(Lt({},Yn),n),Vo(n),r=this.segmentPersonActivation(t,n.internalResolution,n.segmentationThreshold),s=r.segmentation,a=r.heatmapScores,o=r.offsets,i=r.displacementFwd,u=r.displacementBwd,l=r.padding,h=r.internalResolutionHeightAndWidth,c=s.shape,p=c[0],d=c[1],[4,s.data()];case 1:return g=I.sent(),s.dispose(),[4,Kn([a,o,i,u])];case 2:return N=I.sent(),w=N[0],T=N[1],x=N[2],$=N[3],E=Hn(w,T,x,$,this.baseModel.outputStride,n.maxDetections,n.scoreThreshold,n.nmsRadius),E=Xn(E,[p,d],h,l),a.dispose(),o.dispose(),i.dispose(),u.dispose(),[2,{height:p,width:d,data:g,allPoses:E}]}})})},e.prototype.segmentMultiPerson=function(t,n){return n===void 0&&(n=Zn),Vt(this,void 0,void 0,function(){var r,s,a,o,i,u,l,h,c,p,d,g,N,w,T,x,$,E,I,A,F,R=this;return Wt(this,function(k){switch(k.label){case 0:return n=Lt(Lt({},Zn),n),Wo(n),r=on(t),s=r[0],a=r[1],o=jn(n.internalResolution,this.baseModel.outputStride,[s,a]),i=Gn(t,o),u=i.resized,l=i.padding,h=V(function(){var _=R.predictForMultiPersonInstanceSegmentationAndPart(u),b=_.segmentLogits,D=_.longOffsets,P=_.heatmapScores,C=_.offsets,L=_.displacementFwd,q=_.displacementBwd,G=Ce(b,[s,a],o,[[l.top,l.bottom],[l.left,l.right]]),tt;tt=D;var Z=qn(Mt(G),n.segmentationThreshold);return{segmentation:Z,longOffsets:tt,heatmapScoresRaw:P,offsetsRaw:C,displacementFwdRaw:L,displacementBwdRaw:q}}),c=h.segmentation,p=h.longOffsets,d=h.heatmapScoresRaw,g=h.offsetsRaw,N=h.displacementFwdRaw,w=h.displacementBwdRaw,[4,Kn([d,g,N,w])];case 1:return T=k.sent(),x=T[0],$=T[1],E=T[2],I=T[3],A=Hn(x,$,E,I,this.baseModel.outputStride,n.maxDetections,n.scoreThreshold,n.nmsRadius),A=Xn(A,[s,a],o,l),[4,QT(c,p,A,s,a,this.baseModel.outputStride,o,l,n.scoreThreshold,n.refineSteps,n.minKeypointScore,n.maxDetections)];case 2:return F=k.sent(),u.dispose(),c.dispose(),p.dispose(),d.dispose(),g.dispose(),N.dispose(),w.dispose(),[2,F]}})})},e.prototype.segmentPersonPartsActivation=function(t,n,r){var s=this;r===void 0&&(r=.5);var a=on(t),o=a[0],i=a[1],u=jn(n,this.baseModel.outputStride,[o,i]),l=Gn(t,u),h=l.resized,c=l.padding,p=V(function(){var x=s.predictForPersonSegmentationAndPart(h),$=x.segmentLogits,E=x.partHeatmapLogits,I=x.heatmapScores,A=x.offsets,F=x.displacementFwd,R=x.displacementBwd,k=h.shape,_=k[0],b=k[1],D=Ce($,[o,i],[_,b],[[c.top,c.bottom],[c.left,c.right]]),P=Ce(E,[o,i],[_,b],[[c.top,c.bottom],[c.left,c.right]]),C=qn(Mt(D),r);return{partSegmentation:MT(C,P),heatmapScores:I,offsets:A,displacementFwd:F,displacementBwd:R}}),d=p.partSegmentation,g=p.heatmapScores,N=p.offsets,w=p.displacementFwd,T=p.displacementBwd;return h.dispose(),{partSegmentation:d,heatmapScores:g,offsets:N,displacementFwd:w,displacementBwd:T,padding:c,internalResolutionHeightAndWidth:u}},e.prototype.segmentPersonParts=function(t,n){return n===void 0&&(n=Yn),Vt(this,void 0,void 0,function(){var r,s,a,o,i,u,l,h,c,p,d,g,N,w,T,x,$,E;return Wt(this,function(I){switch(I.label){case 0:return n=Lt(Lt({},Yn),n),Vo(n),r=this.segmentPersonPartsActivation(t,n.internalResolution,n.segmentationThreshold),s=r.partSegmentation,a=r.heatmapScores,o=r.offsets,i=r.displacementFwd,u=r.displacementBwd,l=r.padding,h=r.internalResolutionHeightAndWidth,c=s.shape,p=c[0],d=c[1],[4,s.data()];case 1:return g=I.sent(),s.dispose(),[4,Kn([a,o,i,u])];case 2:return N=I.sent(),w=N[0],T=N[1],x=N[2],$=N[3],E=Hn(w,T,x,$,this.baseModel.outputStride,n.maxDetections,n.scoreThreshold,n.nmsRadius),E=Xn(E,[p,d],h,l),a.dispose(),o.dispose(),i.dispose(),u.dispose(),[2,{height:p,width:d,data:g,allPoses:E}]}})})},e.prototype.segmentMultiPersonParts=function(t,n){return n===void 0&&(n=Zn),Vt(this,void 0,void 0,function(){var r,s,a,o,i,u,l,h,c,p,d,g,N,w,T,x,$,E,I,A,F,R,k=this;return Wt(this,function(_){switch(_.label){case 0:return n=Lt(Lt({},Zn),n),Wo(n),r=on(t),s=r[0],a=r[1],o=jn(n.internalResolution,this.baseModel.outputStride,[s,a]),i=Gn(t,o),u=i.resized,l=i.padding,h=V(function(){var b=k.predictForMultiPersonInstanceSegmentationAndPart(u),D=b.segmentLogits,P=b.longOffsets,C=b.heatmapScores,L=b.offsets,q=b.displacementFwd,G=b.displacementBwd,tt=b.partHeatmaps,Z=Ce(D,[s,a],o,[[l.top,l.bottom],[l.left,l.right]]),nt=Ce(tt,[s,a],o,[[l.top,l.bottom],[l.left,l.right]]),wt=P,ot=qn(Mt(Z),n.segmentationThreshold),yt=VT(nt);return{segmentation:ot,longOffsets:wt,heatmapScoresRaw:C,offsetsRaw:L,displacementFwdRaw:q,displacementBwdRaw:G,partSegmentation:yt}}),c=h.segmentation,p=h.longOffsets,d=h.heatmapScoresRaw,g=h.offsetsRaw,N=h.displacementFwdRaw,w=h.displacementBwdRaw,T=h.partSegmentation,[4,Kn([d,g,N,w])];case 1:return x=_.sent(),$=x[0],E=x[1],I=x[2],A=x[3],F=Hn($,E,I,A,this.baseModel.outputStride,n.maxDetections,n.scoreThreshold,n.nmsRadius),F=Xn(F,[s,a],o,l),[4,tE(c,p,T,F,s,a,this.baseModel.outputStride,o,l,n.scoreThreshold,n.refineSteps,n.minKeypointScore,n.maxDetections)];case 2:return R=_.sent(),u.dispose(),c.dispose(),p.dispose(),d.dispose(),g.dispose(),N.dispose(),w.dispose(),T.dispose(),[2,R]}})})},e.prototype.dispose=function(){this.baseModel.dispose()},e}();function SE(e){return Vt(this,void 0,void 0,function(){var t,n,r,s,a,o;return Wt(this,function(i){switch(i.label){case 0:if(t=e.outputStride,n=e.quantBytes,r=e.multiplier,hf==null)throw new Error(`Cannot find TensorFlow.js. If you are using a <script> tag, please also include @tensorflow/tfjs on the page before using this
        model.`);return s=hE(t,r,n),[4,mf(e.modelUrl||s)];case 1:return a=i.sent(),o=new WT(a,t),[2,new If(o)]}})})}function TE(e){return Vt(this,void 0,void 0,function(){var t,n,r,s,a;return Wt(this,function(o){switch(o.label){case 0:if(t=e.outputStride,n=e.quantBytes,hf==null)throw new Error(`Cannot find TensorFlow.js. If you are using a <script> tag, please also include @tensorflow/tfjs on the page before using this
        model.`);return r=cE(t,n),[4,mf(e.modelUrl||r)];case 1:return s=o.sent(),a=new lE(s,t),[2,new If(a)]}})})}function EE(e){return e===void 0&&(e=kf),Vt(this,void 0,void 0,function(){return Wt(this,function(t){return e=vE(e),e.architecture==="ResNet50"?[2,TE(e)]:e.architecture==="MobileNetV1"?[2,SE(e)]:[2,null]})})}function $E(e,t,n,r,s){if(t===void 0&&(t={r:0,g:0,b:0,a:0}),n===void 0&&(n={r:0,g:0,b:0,a:255}),r===void 0&&(r=!1),s===void 0&&(s=[1]),Array.isArray(e)&&e.length===0)return null;var a;Array.isArray(e)?a=e:a=[e];var o=a[0],i=o.width,u=o.height,l=new Uint8ClampedArray(i*u*4);function h(N,w,T,x,$,E){E===void 0&&(E={r:0,g:255,b:255,a:255});for(var I=-1;I<=$;I++)for(var A=-1;A<=$;A++)if(I!==0&&A!==0){var F=(w+I)*x+(T+A);N[4*F+0]=E.r,N[4*F+1]=E.g,N[4*F+2]=E.b,N[4*F+3]=E.a}}function c(N,w,T,x,$,E){$===void 0&&($=[1]),E===void 0&&(E=1);for(var I=0,A=-E;A<=E;A++)for(var F=function(k){if(A!==0&&k!==0){var _=(w+A)*x+(T+k);$.some(function(b){return b===N[_]})||(I+=1)}},R=-E;R<=E;R++)F(R);return I>0}for(var p=0;p<u;p+=1)for(var d=function(N){var w=p*i+N;l[4*w+0]=n.r,l[4*w+1]=n.g,l[4*w+2]=n.b,l[4*w+3]=n.a;for(var T=function($){if(s.some(function(I){return I===a[$].data[w]})){l[4*w]=t.r,l[4*w+1]=t.g,l[4*w+2]=t.b,l[4*w+3]=t.a;var E=c(a[$].data,p,N,i,s);r&&p-1>=0&&p+1<u&&N-1>=0&&N+1<i&&E&&h(l,p,N,i,1)}},x=0;x<a.length;x++)T(x)},g=0;g<i;g+=1)d(g);return new ImageData(l,i,u)}const Zt=document.createElement("canvas");Zt.id="canvas";Zt.style.display="none";document.body.appendChild(Zt);const Zr=Zt.getContext("2d");let xf=null;async function _E(){xf=await EE({architecture:"MobileNetV1",outputStride:16,multiplier:.75,quantBytes:2})}_E();window.processImage=async function(e){const t=new Image;t.crossOrigin="anonymous",t.onload=async()=>{Zt.width=t.width,Zt.height=t.height,Zr.clearRect(0,0,Zt.width,Zt.height),Zr.drawImage(t,0,0);const n=await xf.segmentPersonParts(t,{internalResolution:"medium",segmentationThreshold:.7}),s=$E(n,{r:0,g:255,b:0,a:255},{r:0,g:0,b:0,a:0},[11,12,13,14,15,16,23,24]);Zr.putImageData(s,0,0);const a=Zt.toDataURL("image/png");window.prompt(a,"IMAGE_DATA")},t.src=e};
