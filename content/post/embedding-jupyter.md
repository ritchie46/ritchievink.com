+++
date = "2019-03-17"
description = "Embedding jupyter notebook in an iframe and serve as a reverse proxy behind NGINX"
tags = ["save-some-time", "web", "python"]
draft = false
keywords =["web", "python"]
author = "Ritchie Vink"
title = "Save some time: Embedding jupyter notebook in an iframe and serve as a reverse proxy behind NGINX" 
og_image = "/img/post-22-embedding-jupyter/crossroads.jpg"
+++

{{< figure src="/img/post-22-embedding-jupyter/crossroads.jpg" >}}

<br>
Embedding Jupyter notebook/ lab on your website can be done by embedding it in an iframe. However, it takes some configurational quirks to get it done. For my purpose, I also needed to offload validation to another service on the backend. Both the validation server as the jupyter notebook server were proxied behind an NGINX server. Here is the configuration.

## NGINX setup
In the configuration, we set two upstream servers. One is the actual jupyter notebook server and one is the validation server. The `/notebook` endpoint is exposed to the web. The `/auth` endpoint is only internally visible and is used for incoming request authentication.

Normally an NGINX reverse proxy server uses the HTTP protocol. However jupyter 
notebook requires a websocket connection to work properly. By setting the **Upgrade** and the **Connection** header we can create a websocket tunnel between client and server. We also want this connection to remain open. Therefore we set `proxy_read_timeout` to a large value.


``` text
http {

    upstream notebook {
        # The notebook server
        server 127.0.0.1:8888;
    }

    upstream auth-server {
        # whatever authentication method your setup requires
        server 127.0.0.1:8000;
    }

    server {
        listen 80;

        location /notebook {
            # validate the request with the /auth endpoint
            auth_request       /auth;

            # Allow iframe inbedding from this parent <your-website.com>
            add_header Content-Security-Policy "frame-ancestors http://<your-website.com>:80";
            
            # websocket proxy
            proxy_http_version 1.1;
            proxy_redirect off;
            proxy_buffering off;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_read_timeout 86400;

            proxy_pass         http://notebook;
        }

        location = /auth {
            internal;
            proxy_pass                http://auth-server/;
            # body content not needed
            proxy_pass_request_body   off;
            proxy_set_header          Content-Length "";
            proxy_set_header          X-Original-URI $request_uri;
        }
    }

```

## Jupyter 
Below is the configuration of `jupyter_notebook_config.py` shown. The important part for embedding the notebook in the iframe is `c.NotebookApp.tornado_settings`. Here you set URL of the iframe's parent, i.e. your website. 

``` python
c.NotebookApp.port = 8888
c.NotebookApp.token = ''

## The IP address the notebook server will listen on.
c.NotebookApp.ip = '*'
c.NotebookApp.base_url = 'notebook'
c.NotebookApp.allow_origin = '*'
c.NotebookApp.open_browser = False
```

## HTML

Finally, you need to point the iframe to the proper endpoint and you are all set.

``` HTML
<h1>Welcome to my cool website</h1>
<iframe src="/notebook"></iframe>
```

## Save some time
**Save some time** posts are short write ups with the intention to, well, save you some time. The content can be anything that has cost me more time it should have.
