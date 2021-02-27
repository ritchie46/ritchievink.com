+++
date = "2021-02-27"
description = "Design decisions of the Polars dataframe libary"
tags = ["computer science", "python", "rust"]
draft = false
author = "Ritchie Vink"
title = "Polars working title"
keywords = []
og_image = "/img/post-33-async-tasks/queue.png"
+++


{{< figure src="/img/post-33-async-tasks/queue.png" >}}
<br>    

## 1. Introduction
At the time of writing this, the coronavirus has been in our country for exact a year, which means I have been sitting at home for a very long time. At the start of the pandemic I had a few pet project in Rust under my belt and I noticed that the *"are we DataFrame yet"*, wasn't anywhere near my satisfaction, so I wondered if I could make minimalistic crate that solved a work use case of mine. But boy, did that get out of hand. 

It has been a year with lots of programming resulting in one of the fastest DataFrame libaries available. This this is my first official "Hello World" from [polars](https://github.com/ritchie46/polars) on my personal blog. With this post I hope I can take the reader along with some design decisions I encountered and get a more thorough understanding how Polars works.

## 2. Know your hardware
If you want to design for optimal performance you cannot ignore hardware. There are cases where algorithmic complexity doesn't give you a good intuition of the real performance due to hardware related issue like cache hierarchies and branch prediction. For instance, up to a certain number of elements (a few 100, depending on the datatype) it is faster to lookup a given element in an array than to look it up in hashmap, whereas the time complexity of those data structures are $ \mathcal{O}(n) $, and $ \mathcal{O}(1) $ respectively. 
This makes design decision also very temporal, what is a bottleneck in this day, may not be in the future. This is clearly seen in database systems. DB systems from the previous generation like PostgreSQL or MySQL are all row based volcano models<sup>[1]</sup>, which was an excellent design decision in that era when disks were much slower and RAM memory was very limited. Nowadays we have fast SSD disks and large amounts of memory available, and wide SIMD registers, we see that columnar databases like cockroachDB, DuckDB are among the best performing DBMSes.

### 2.1 Cache hierarchies
Oversimplifying, RAM memory comes in two flavors, large and slow or fast and small. For this reason you have memory caches in a hierarchy. You've got main memory that's large and slow. And memory you've used last is stored in L1, L2, L3 cache with more latency respectively. The summation below gives you and idea of the relative latency of the different cache levels:

**Access time in CPU cycles**
* CPU register - 1 cycle
* L1 cache - ~1-3 cycles
* L2 cache - ~10 cycles
* Main memory - ~250 cycles

When accessing data sequentially we want to make sure that data is in cache as much as possible, or we could easily have a ~100x performance penalty. Caches are loaded and deleted in cache lines. When we load a single data point, we get a whole cache line, but we also remove a whole cache line. They are typically 64, or 128 bytes long and aligned on 64 byte memory adresses.

### 2.2 Prefetching and branch predictions
CPU's prefetch data and instructions to local cache to reduce the high penalty of memory latency. If a you have a tight loop without any branches (if-else-then structures) the CPU has no problem knowing which data to prefetch and can fully utilize [instruction pipelines](https://en.wikipedia.org/wiki/Instruction_pipelining). 

Instruction pipelines hide latency by doing work in parallel. Every CPU instruction goes through a **Fetch, Decode, Execute, Write-back** sequence. Instead of doing these 4 instruction sequantially in a single pipeline, there are multiple pipelines that already pre-fetch (decode, execute, etc.) the next instructions. This increases throughput and hides latency. However if this process is interupted, you start with empty pipelines and you have to wait the full latency period for the next instruction. Below we see a visual of instruction pipelines.

{{< figure src="/img/post-34-polars/4stage_pipe_line.png" title="4 stage instruction pipeline" >}}

The CPU does it's best to predict which conditional jump is taken and speculatively execute that code in advance (i.e. keep the pipelines filled), but if it has mispredicted it must clear that work and we pay the latency price until the pipelines are full again. 

### 2.3 SIMD instructions
Modern processors have SIMD registers (Single Instruction Multiple Data), which operate on whole vectors of data in a single CPU cycle. These register greatly improve the performance of simple operations if your data is linear in memory. A columnar memory format can therefore fully utilize SIMD instructions. Polars and its memory backend Arrow, utilize SIMD to get optimal performance.

{{< figure src="/img/post-34-polars/SIMD.jpeg" title="SIMD" >}}

## 3. Arrow memory format
Polars is based the Rust native implementation [Apache Arrow](https://github.com/apache/arrow). Arrow can be seen as middleware software for DBMS, query engines and DataFrame libraries. Arrow provides very cache-coherent data structures and proper missing data handling.

Let's go through some examples to see what this means.

### 3.1 Arrow numeric array
An Arrow numeric array consists of a data buffer containing some typed data, e.g. `f32`, `u64`, etc. , shown in the figure as orange array. Besided the value data, an Arrow array alwas has a validity buffer. This buffer is a bit array where the bits indicate missing data. Because the missing data is represented by bits there is minimal memory overhead.

This directly shows a clear advantage over Pandas for instance, where there is no clear distinciton between a float `NaN` and missing data, where they really should represent different things. 

{{< figure src="/img/post-34-polars/arrow_primitive.svg" title="Arrow numeric array" >}}

### 3.1 Arrow string array
The figure below shows the memory layout of an Arrow `LargeString` array. This figure encodes the following array `["foo", "bar", "ham"]`. The Arrow array consists of a data buffer where all string bytes are concatenated to a single sequantial buffer (good for cache coherence!). To be able to find the starting and ending position of a string value there is a seperate offset array, and finally there is the null-bit buffer to indicate missing values.

{{< figure src="/img/post-34-polars/arrow_string.svg" title="Arrow large-utf8 array" >}}

Let's compare this with a pandas string array. Pandas strings are actually Python objects, therefore they are boxed (which means there is also memory overhead to encode the type next to the data). Sequential string access in pandas will lead to cache miss after cache miss, because every string value may point to a completely different memory location. 

{{< figure src="/img/post-34-polars/pandas_string.svg" title="Pandas string array" >}}

For cache coherence the Arrow representation is a clear winner. However, there is a price. If we want to filter this array or we want to take values based on some index array we need to copy a lot more data around. The pandas string array only holds pointers to the data and can cheaply create a new array with pointers. Arrow string arrays have to copy all the string data, especially when you have large string values this can become a very large overhead. It is also harder to estimate the size of the string data buffer, as this comprises the length of all string values. 

Polars also has a `Categorical` type which helps you mitigate this problem. Arrow also has a solution for this problem, called the `Dictionary` type, which is similar to Polars' `Categorical` type.

### 3.3 Reference counted
Arrow buffers are reference counted and immutable. Meaning that copying a DataFrame, Series, Array is almost a no-op, making it very easy to write pure functional code. The same counts for slicing operations, which are only an increment of the reference count and a modification of the offset.

### 3.2 Performance: Missing data and branching
As we've seen, the missing data is encoded in a seperate buffer. This means we can easily write branchless code by just ignoring the null buffer during an operation. When the operation is finished the null bit buffer is just copied to the new array. When a branch miss would be more expensive than the execution of a operation this is an easy win and used in many operations in Arrow and Polars.

### 3.3 Performance: filter-trick
An operation that has to be done often in a DBMS, is a filter. Based on some predicate (boolean mask) we filter rows. Arrows null bit buffer allows for very fast filtering using a **filter-trick** (unoffically named by me). Note, that this filter-trick often leads to faster filters, but it may not always be the case. If your predicate consists alternating boolean values e.g. <br>
`[true, false, true, false, ... , true, false]` this trick has slight overhead. 

The core idea of the filter-trick is that we can load the bit array from memory as any integer type we want. Let's say that we load the bit array as an unsigned integer `u64` then we know that the maximum encoded value is $2^{64}$ (64 consecutive one value in binary), and the minimum encoded value is 0 (64 consecutive 0 values in binary). We can make a table of 64 entries that represent how many consecutive values we can filter and skip. If this integer is in this table we know have many values to filter in very few CPU cycles and can do a `memcpy` to efficiently copy data. If the integer is not present in the table we have to iterate through the bits one by one and hope for a hit in the following 64 bits we load.

{{< figure src="/img/post-34-polars/filter-trick.svg" title="filter-trick" >}}

This filter-trick is used of course in any operation that includes a predicate mask, such as `filter`, `set` and `zip`.

## 4. Parallelization
With the plateauing of CPU clock speeds and the end of Moore's law in sight, the free lunch <sup>[2]</sup> is over. Single threaded performance isn't going to increase much anymore. To mitigate this almost all hardware nowadays has multiple cores. My laptop has 12 logical cores, so there is an tremendous potential for paralellization. Polars is written to exploit parallelism as much as possible. 

### 4.1 Embarrasingly parallel
The best parallelization is of course where is no need for communication and there are no data dependencies. This is for instance the case if we do an aggregation on the columns in a DataFrame. All columns can be aggregated on in parallel. 

Another embarissangly parallel algorithm in Polars in the apply phase in a groupy-operation.

{{< figure src="/img/post-34-polars/split-apply-combine-par.svg" title="Embarrassingly parallel apply phase" >}}

### 4.2 Parallel hashing



## References
&nbsp; [1] Graefe G.. Volcano (1994) *an extensible and parallel query evaluation system.* [IEEE Trans. Knowl. Data Eng.](https://paperhub.s3.amazonaws.com/dace52a42c07f7f8348b08dc2b186061.pdf)<br>
&nbsp; [2] Herb Sutter (2005) *The Free Lunch Is Over: A Fundamental Turn Toward Concurrency in Software* [Weblog](http://www.gotw.ca/publications/concurrency-ddj.htm)


## Last words
Lorem ipsum

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]},
    TeX: { equationNumbers: { autoNumber: "AMS" } }
  });
  </script>



<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.6/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

<head>

<style>

.formula-wrap {
overflow-x: scroll;
}

</style>

</head>
