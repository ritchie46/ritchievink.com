+++
date = "2021-02-28"
description = "Design decisions of the Polars dataframe libary"
tags = ["computer science", "python", "rust"]
draft = false
author = "Ritchie Vink"
title = "I wrote one of the fastest DataFrame libraries"
keywords = []
og_image = "/img/post-34-polars/polars.webp"
+++


{{< figure src="/img/post-34-polars/polars.webp"  >}}
<br>    

## 1. Introduction
At the time of writing this, the coronavirus has been in our country for a year, which means I have been sitting at home for a very long time. At the start of the pandemic, I had a few pet projects in Rust under my belt and I noticed that the *"are we DataFrame yet"*, wasn't anywhere near my satisfaction. So I wondered if I could make a minimalistic crate that solved a specific use case of mine. But boy, did that get out of hand. 

A year later with lots of programming resulting in one of the fastest DataFrame libraries available in Rust and Python. This is my first official "Hello World" from [polars](https://github.com/ritchie46/polars) on my personal blog. With this post, I hope I can take the reader along with some design decisions I encountered and get a more thorough understanding of how Polars works under the hood.

## 2. Quite a bold claim
I know it is quite a bold claim to make, and I would not make it lightly. There is a benchmark for database systems that does a benchmark on in-memory tools ran by h2o.ai. This benchmark consists of 10 groupby tests on different data cardinalities and query complexity to give a well-rounded view of a tool's performance, and 5 tests on different join questions. At the time of writing this blog, Polars is the fastest DataFrame library in the benchmark second to R's data.table, and Polars is top 3 all tools considered.

Below are shown the summaries of the 5GB dataset test, and you can see the whole [benchmark here](https://h2oai.github.io/db-benchmark/).

<div style="max-width: 60%">
{{< figure src="/img/post-34-polars/db-benchmark-join-5GB.png" title="Join benchmark summary" >}}
{{< figure src="/img/post-34-polars/db-benchmark-gb-simple-5GB.png" title="Groupby benchmark summary (basic questions)" >}}
{{< figure src="/img/post-34-polars/db-benchmark-gb-adv-5GB.png" title="Groupby benchmark summary (advanced questions)" >}}

</div>

## 2. Know your hardware
If you want to design for optimal performance you cannot ignore hardware. There are cases where algorithmic complexity doesn't give you a good intuition of the real performance due to hardware-related issues like cache hierarchies and branch prediction. For instance, up to a certain number of elements (a few 100, depending on the datatype), it is faster to lookup a given element in an array than to look it up in hashmap, whereas the time complexity of those data structures are $ \mathcal{O}(n) $, and $ \mathcal{O}(1) $ respectively. 
This makes design decisions also very temporal, what is a bottleneck in this day, may not be in the future. This is clearly seen in database systems. DB systems from the previous generation like PostgreSQL or MySQL are all row-based volcano models<sup>[1]</sup>, which was an excellent design decision in that era when disks were much slower and RAM was very limited. Nowadays we have fast SSD disks and large amounts of memory available, and wide SIMD registers, we see that columnar databases like cockroachDB, DuckDB are among the best performing DBMSes.

### 3.1 Cache hierarchies
Oversimplifying, RAM comes in two flavors, large and slow or fast and small. For this reason, you have memory caches in a hierarchy. You've got main memory that's large and slow. And memory you've used last is stored in L1, L2, L3 cache with more latency respectively. The summation below gives you an idea of the relative latency of the different cache levels:

**Access time in CPU cycles**
* CPU register - 1 cycle
* L1 cache - ~1-3 cycles
* L2 cache - ~10 cycles
* Main memory - ~250 cycles

When accessing data sequentially we want to make sure that data is in cache as much as possible, or we could easily have a ~100x performance penalty. Caches are loaded and deleted in cache lines. When we load a single data point, we get a whole cache line, but we also remove a whole cache line. They are typically 64, or 128 bytes long and aligned on 64-byte memory adresses.

### 3.2 Prefetching and branch predictions
CPUs prefetch data and instructions to a local cache to reduce the high penalty of memory latency. If you have a tight loop without any branches (if-else-then structures) the CPU has no problem knowing which data to prefetch and can fully utilize [instruction pipelines](https://en.wikipedia.org/wiki/Instruction_pipelining). 

Instruction pipelines hide latency by doing work in parallel. Every CPU instruction goes through a **Fetch, Decode, Execute, Write-back** sequence. Instead of doing these 4 instructions sequentially in a single pipeline, there are multiple pipelines that already pre-fetch (decode, execute, etc.) the next instructions. This increases throughput and hides latency. However, if this process is interrupted, you start with empty pipelines and you have to wait the full latency period for the next instruction. Below we see a visual of instruction pipelines.

<div style="max-width: 50%">{{< figure src="/img/post-34-polars/4stage_pipe_line.png" title="4 stage instruction pipeline" >}}</div>

The CPU does its best to predict which conditional jump is taken and speculatively execute that code in advance (i.e. keep the pipelines filled), but if it has mispredicted it must clear that work and we pay the latency price until the pipelines are full again. 

### 3.3 SIMD instructions
Modern processors have SIMD registers (Single Instruction Multiple Data), which operate on whole vectors of data in a single CPU cycle. The vector lane widths vary from 128 bits to 512 bits, and the speedup depends on the width of the registers and the number of bits needed to represent the datatype. These register greatly improve the performance of simple operations if you can fill them fast enough (linear data). A columnar memory format can therefore fully utilize SIMD instructions. Polars and its memory backend Arrow, utilize SIMD to get optimal performance.

{{< figure src="/img/post-34-polars/SIMD.svg" title="SISD vs SIMD" >}}

## 4. Arrow memory format
Polars is based on the Rust native implementation [Apache Arrow](https://github.com/apache/arrow). Arrow can be seen as middleware software for DBMS, query engines and DataFrame libraries. Arrow provides very cache-coherent data structures and proper missing data handling.

Let's go through some examples to see what this means.

### 4.1 Arrow numeric array
An Arrow numeric array consists of a data buffer containing some typed data, e.g. `f32`, `u64`, etc. , shown in the figure as an orange colored array. Besides the value data, an Arrow array alwas has a validity buffer. This buffer is a bit array where the bits indicate missing data. Because the missing data is represented by bits there is minimal memory overhead.

This directly shows a clear advantage over Pandas for instance, where there is no clear distinction between a float `NaN` and missing data, where they really should represent different things. 

{{< figure src="/img/post-34-polars/arrow_primitive.svg" title="Arrow numeric array" >}}

### 4.1 Arrow string array
The figure below shows the memory layout of an Arrow `LargeString` array. This figure encodes the following array `["foo", "bar", "ham"]`. The Arrow array consists of a data buffer where all string bytes are concatenated to a single sequential buffer (good for cache coherence!). To be able to find the starting and ending position of a string value there is a separate offset array, and finally, there is the null-bit buffer to indicate missing values.

{{< figure src="/img/post-34-polars/arrow_string.svg" title="Arrow large-utf8 array" >}}

Let's compare this with a pandas string array. Pandas strings are actually Python objects, therefore they are boxed (which means there is also memory overhead to encode the type next to the data). Sequential string access in pandas will lead to cache miss after cache miss, because every string value may point to a completely different memory location. 

{{< figure src="/img/post-34-polars/pandas_string.svg" title="Pandas string array" >}}

For cache coherence, the Arrow representation is a clear winner. However, there is a price. If we want to filter this array or we want to take values based on some index array we need to copy a lot more data around. The pandas string array only holds pointers to the data and can cheaply create a new array with pointers. Arrow string arrays have to copy all the string data, especially when you have large string values this can become a very large overhead. It is also harder to estimate the size of the string data buffer, as this comprises the length of all string values. 

Polars also has a `Categorical` type which helps you mitigate this problem. Arrow also has a solution for this problem, called the `Dictionary` type, which is similar to Polars' `Categorical` type.

### 4.3 Reference counted
Arrow buffers are reference counted and immutable. Meaning that copying a DataFrame, Series, Array is almost a no-op, making it very easy to write purely functional code. The same counts for slicing operations, which are only an increment of the reference count and a modification of the offset.

### 4.2 Performance: Missing data and branching
As we've seen, the missing data is encoded in a separate buffer. This means we can easily write branchless code by just ignoring the null buffer during an operation. When the operation is finished the null bit buffer is just copied to the new array. When a branch miss would be more expensive than the execution of an operation this is an easy win and used in many operations in Arrow and Polars.

### 4.3 Performance: filter-trick
An operation that has to be done often in a DBMS, is a filter. Based on some predicate (boolean mask) we filter rows. Arrows null bit buffer allow for very fast filtering using a **filter-trick** (unofficially named by me). _*Credits to the filter-trick go the Apache Arrow implementation_. Note, that this filter-trick often leads to faster filters, but it may not always be the case. If your predicate consists of alternating boolean values e.g. <br>
`[true, false, true, false, ... , true, false]` this trick has a slight overhead. 

The core idea of the filter-trick is that we can load the bit array from memory as any integer type we want. Let's say that we load the bit array as an unsigned integer `u64` then we know that the maximum encoded value is $2^{64}$ (64 consecutive one value in binary), and the minimum encoded value is 0 (64 consecutive 0 values in binary). We can make a table of 64 entries that represent how many consecutive values we can filter and skip. If this integer is in this table we know have many values to filter in very few CPU cycles and can do a `memcpy` to efficiently copy data. If the integer is not present in the table we have to iterate through the bits one by one and hope for a hit in the following 64 bits we load.

{{< figure src="/img/post-34-polars/filter-trick.svg" title="filter-trick" >}}

This filter-trick is used of course in any operation that includes a predicate mask, such as `filter`, `set`, and `zip`.

## 5. Parallelization
With the plateauing of CPU clock speeds and the end of Moore's law in sight, the free lunch <sup>[2]</sup> is over. Single-threaded performance isn't going to increase much anymore. To mitigate this, almost all hardware nowadays has multiple cores. My laptop has 12 logical cores, so there is a tremendous potential for parallelization. Polars is written to exploit parallelism as much as possible. 

### 5.1 Embarrasingly parallel
The best parallelization is of course where is no need for communication and there are no data dependencies. Polars utilizes this kind of parallelism as much as possible. 

This is for instance the case if we do an aggregation on the columns in a DataFrame. All columns can be aggregated on in parallel. 

Another embarrassingly parallel algorithm in Polars in the apply-phase in a groupy-operation.

{{< figure src="/img/post-34-polars/split-apply-combine-par.svg" title="Embarrassingly parallel apply phase" >}}

### 5.2 Parallel hashing
Hashing is the core of many operations in a DataFrame library, a groupby-operation creates a hash table with the group index pointers, and a join operation needs a hash table to find the tuples mapping the rows of the left to the right DataFrame. 

#### 5.2.1 Expensive synchronization
In both operations, we cannot simply split the data among the threads. There is no guarantee that all the same keys would end up in the same hash table on the same thread. Therefore we would need an extra synchronization-phase where we build a new hashtable. This principle is shown in the figure below for 2 threads.

{{< figure src="/img/post-34-polars/naive-hash.svg" title="Expensive synchronization" >}}

#### 5.2.2 Expensive locking
Another option that is found too expensive is hashing the data on separate threads and have a single hash table in a `mutex`. As you can imagine, thread contention is very high in this algorithm and the parallelism doesn't really pay of.

{{< figure src="/img/post-34-polars/lock-hash.svg" title="Expensive locking" >}}

#### 5.2.2 Lock-free hashing
Instead of the before mentioned approaches, Polars uses a lock-free hashing algorithm. This approach does do more work than the previous *Expensive locking* approach, but this work is done in parallel and all threads are guaranteed to not have to wait on any other thread. Every thread computes the hashes of the keys, but depending on the outcome of the hash, it will determine if that key belongs to the hash table of that thread. This is simply determined by the `hash value % thread number`. Due to this simple trick, we know that every threaded hash table has unique keys and we can simply combine the pointers of the hash tables on the main thread.

{{< figure src="/img/post-34-polars/lock-free-hash.svg" title="Lock-free hashing" >}}

## 6. Query optimization: Less is more
The best performance gains are simply not doing any work at all. Polars consists of two public APIs, one that is eager, procedural programming, and one that is lazy declarative programming. I would recommend using the lazy API as much as possible when dealing with performance-critical code. 

The declarative DSL allows Polars to analyze a logical plan of your query and it can apply several optimization/ heuristics such that your query can be executed by just doing less work. If you want to read more about the optimizations that are done to the query plan, there is a [section with examples in the polars book](https://ritchie46.github.io/polars-book/lazy_polars/predicate_pushdown.html).

## 7. And more
This post only highlighted a few of the performance related designs in the Polars and Arrow library. Other things that are implemented are for instance:

* Ad hoc partitioning (map-reduce like groupby's)
* [Vectorized hashing](https://www.cockroachlabs.com/blog/vectorized-hash-joiner/) <sup>[3]</sup> (Tipped by [Daniël Heres](https://github.com/Dandandan)).
* Using table statistics to choose fast paths
* And much more..

Now the "hello world" is officially out there, I may highlight those other subjecst in later posts. Check out [polars on github](https://github.com/), and if you have any remarks, feature requests, etc. let me know!


## References
&nbsp; [1] Graefe G.. Volcano (1994) *an extensible and parallel query evaluation system.* [IEEE Trans. Knowl. Data Eng.](https://paperhub.s3.amazonaws.com/dace52a42c07f7f8348b08dc2b186061.pdf)<br>
&nbsp; [2] Herb Sutter (2005) *The Free Lunch Is Over: A Fundamental Turn Toward Concurrency in Software* [Weblog](http://www.gotw.ca/publications/concurrency-ddj.htm)
&nbsp; [3] Angela Chang: CockroachDB (2019) *40x faster hash joiner with vectorized execution* [Weblog](https://www.cockroachlabs.com/blog/vectorized-hash-joiner/)



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
