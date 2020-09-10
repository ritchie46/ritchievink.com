+++
date = "2020-09-10"
description = "Asynchronous task handling in Rust"
tags = ["save some time", "rust"]
draft = false
author = "Ritchie Vink"
title = "Asynchronous task handling in Rust"
keywords = []
og_image = "/img/post-33-async-tasks/queue.png"
+++


{{< figure src="/img/post-33-async-tasks/queue.png" >}}
<br>    

To prevent server timeouts and/or to keep things snappy, I often find myself needing a tool like [Celery](https://docs.celeryproject.org/en/stable/getting-started/introduction.html) to offload tasks asynchronously. These tasks most of the time have requirements that make them hard to centralize, resulting in multiple services in k8 that have their own celery workers in the pod. This feels like overkill, especially as a lot of tasks are really lightweight. This post shows how we can set up a simple async task queue in Rust and the tokio runtime.

## Setup
Let's start by setting up a new crate and defining the dependencies. `$ cargo new async_tasks && cd async_tasks && rustup override set nightly`. At the time of writing [Rocket](https://rocket.rs/) still requires nightly to compile, thus we use nightly.

Next we'll add the following dependencies to `Cargo.toml`

```toml
[dependencies]
rocket = "0.4.5"
tokio = { version = "0.2.22", features = ["rt-core", "sync", "time"], default-features = false }
```

## A simple server
Let's start with a very simple server. The contents of `src/main.rs` are:

```rust
#![feature(decl_macro)]
#[macro_use]
extern crate rocket;

#[get("/")]
fn index() -> &'static str {
    "Hello, world!"
}

fn main() {
    rocket::ignite().mount("/", routes!(index)).launch();
}
```

If we run this setup with `$cargo run`, we can check if all works well in another terminal: `$ curl http://localhost:8000` should output:

```text
Hello, world!
```

## Tasks and queues

Ok, let's extend this basic setup with asynchronous tasks. 
Before we'll dive deeper let's describe the setup I want. 
I want my endpoints to be able to define **synchronous** and **asynchronous** tasks. These tasks will be added to a **queue**. On the other side of this **queue**. There is only a single consumer of the queue, but all endpoints may define a new task, hence there are multiple producers, so we will use an `mpsc` channel as a queue.

Let's start with the skeleton. Every _async_ code needs an _async_ runtime that handles which piece of code get's execution time.

```rust
use tokio::runtime::{
    Runtime,
    Builder
};

/// A simple single threaded runtime
pub fn simple_async_runtime() -> std::io::Result<Runtime> {
    Builder::new().enable_all().build()
}
```

Just before we start `rocket` we spawn a new thread that will handle all the async tasks. This thread will have the receiving end `rx` or the `mpsc::channel`.

```rust
fn main() {
    let (tx, rx) = mpsc::channel(32);

    // this thread will fetch tasks from the channels receiving end and execute them
    std::thread::spawn(|| {
        let mut rt = simple_async_runtime().unwrap();
        rt.block_on(task_exec(rx))
    });

    rocket::ignite().mount("/", routes!(index)).launch();
}
```

This thread blocks until `task_exec` is finished (which won't as we later see).

```rust
use tokio::{
    runtime::{Builder, Runtime},
    sync::mpsc::{self, Receiver, Sender},
};

// Temorary task. We'll define this later
type Task = ();

async fn task_exec(mut rx: Receiver<Task>) {
    todo!()
}
```

Let's give a little more body to the `task_exec` function. Before we do this I will expand the `Task` type into an enum differentiating between sync and async tasks, as I can imagine that we want to be able to define tasks that run synchronously.

```rust
type SyncTask = ();
type AsyncTask = ();

enum Task {
    Sync(SyncTask),
    Async(AsyncTask)
}

async fn task_exec(mut rx: Receiver<Task>) {
    println!("task executer started");
    // loop forever
    loop {
        if let Some(task) = rx.recv().await {
            println!("start new task");
            use Task::*;
            match task {
                Sync(t) => todo!(),
                Async(t) => todo!()
            }
        }
    }
}
```

In the `if let Some(task) = rx.recv().await` line we fetch a new task form the queue. This task will be executed sync or async depending on its type. We haven't yet defined what these tasks are. However, I'd like the `task_exec` function to be agnostic of the type and be able to run any task. 
This can be done by using closures. Closures can have a very simple type `Fn() -> ()`. This may seem limiting as this function doesn't take any arguments, but with closures, we don't need any!  

Closures can capture any number of arguments from their lexical scope whilst maintaining this simple `Fn() -> ()` type.

In code, the actual type is slightly more complex but bear with me.

```
use std::{future::Future, pin::Pin};

type AsyncTask = Box<dyn (FnOnce() -> Pin<Box<dyn Future<Output = ()>>>) + Send>;
type SyncTask = Box<dyn FnOnce() + Send>;
```

Let's start with the `SyncTask`. We `Box` the closure as we cannot know statically (during compilation) what the size of our closure is. Therefore we need to make a heap allocation that hides our closure behind a pointer. Next, we add the `Send` trait because we want our closure to be sendable between threads (from the rocket endpoints to the worker thread). 

The `AsyncTask` is boxed for the same reason. However, this task is **async** and therefore does have a return type. Every time `.await` is called, this closure returns a type that implements the `Future` trait. We don't know which types will implement this `Future` trait and we want to be able to use different tasks (with different `Future` implementations), hence we also `Box` the return type.

Lastly, the return type is `Pin`, meaning that its location in memory is fixed. This allows the async runtime to continue where it's left off and having the guarantee that all pointers are valid. [Read more about pinning in relation to async/await](https://rust-lang.github.io/async-book/04_pinning/01_chapter.html).

With those types in place, let's finish the `task_exec` function:

```rust
async fn task_exec(mut rx: Receiver<Task>) {
    println!("task executer started");
    // loop forever
    loop {
        if let Some(task) = rx.recv().await {
            println!("start new task");
            use Task::*;
            match task {
	    	// call the closure sync
                Sync(t) => t(),
		// call the closure async
                Async(t) => t().await
            }
        }
    }
}
```

## Wrap it up
Now we almost have everything in place to create and execute the tasks. Let's start with a function that will initiate the tasks synchronously. This function `run_task` will be called from any rocket endpoint and will block until the _task/message_ is sent to the worker thread.

```rust
/// Send a task to the worker. This code blocks until the message is sent (which should be fast)
fn run_task(task: Task, mut tx: Sender<Task>) {
    let mut rt = simple_async_runtime().unwrap();

    // this blocks only the sending not the execution of the task
    rt.block_on(async move {
        // should not fail as the executing thread should remain alive.
        // note: could not use unwrap() or except() because the Error didn't implement Debug
        assert!((tx.send(task).await.is_ok()));
    });
    println!("task sent")
}
```

There is only one last thing needed before we can write our endpoint that will fill the task queue. The endpoints need to get a hold on the sending side of the `mpsc::channel`. This can be achieved by attaching this channel side `tx` to Rocket before we launch the rocket😁.

```rust
    rocket::ignite()
    	// attach the sender to rocket
        .manage(tx)
        // already added all endpoints
        .mount("/", routes!(index, sync_task_endpoint, async_task_endpoint))
        .launch();
```

### A sync task
All we need is an endpoint to start a synchronous task (sync in the worker thread, not in the endpoint). We define the task closure in the endpoint. The task will block the thread for 3 seconds before it is finished.

Note that we use the `move` keyword to take ownership of `some_var` this can be used to start the same task with different arguments.

```rust
#[get("/start_sync_task")]
fn sync_task_endpoint(tx: State<Sender<Task>>) -> &'static str {
    let some_var = "foo";
    let task = move || {
        println!("start sync task with {}", some_var);
        std::thread::sleep(std::time::Duration::from_secs(3));
        println!("sync task finished");
    };

    run_task(Task::Sync(Box::new(task)), tx.inner().clone());
    "task started"
}
```

### An async task
An async task isn't much more complicated at this point. It only needs an async task and a carefully placed `Box::pin`

```rust
#[get("/start_async_task")]
fn async_task_endpoint(tx: State<Sender<Task>>) -> &'static str {
    let some_var = "foo";

    // we need to help the compiler a bit with the type
    let task: AsyncTask = Box::new(move || {
        Box::pin(async move {
            println!("start async task with {}", some_var);
            tokio::time::delay_for(std::time::Duration::from_secs(3)).await;
            println!("sync atask finished");
            ()
        })
    });

    run_task(Task::Async(task), tx.inner().clone());
    "task started"
}
```

### Validation

`$ cargo check` is happy, but let's see it it really works. Spin up the server with `$ cargo run`, and again from another terminal run `$ curl http://localhost:8000/start_sync_task`

**Rocket output:**
```text
GET /start_sync_task:
    => Matched: GET /start_sync_task (sync_task_endpoint)
task sent
start new task
start sync task with foo
    => Outcome: Success
    => Response succeeded.
sync task finished
```

And for the async task `$ curl http://localhost:8000/start_async_task`:

**Rocket output:**
```text
GET /start_async_task:
    => Matched: GET /start_async_task (async_task_endpoint)
task sent
start new task
start async task with foo
    => Outcome: Success
    => Response succeeded.
async task finished
```

And that's our validation. Both responses finish before the tasks have.

## Last words
This is of course not a drop-in replacement of celery or other more complicated task queues. For instance, we haven't build anything to poll the status of our tasks, nor is there any distribution. However, it may be a good lightweight starting template for simple tasks in your web services. You can find the complete code in this [repo](https://github.com/ritchie46/async_task_template).


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
