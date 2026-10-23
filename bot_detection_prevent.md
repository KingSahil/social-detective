# Bot Detection Mechanisms

When a Python script (whether using `requests` or `playwright`) hits a protected endpoint, it goes through several layers of deep packet inspection and client-side interrogation by WAFs like Cloudflare, DataDome, or Akamai.

### Layer 1: Network & Transport Layer (TCP/IP)
Before an HTTP request is even formed, the WAF analyzes the TCP connection.
* **IP Reputation & ASN:** The WAF checks the Autonomous System Number (ASN) of your IP. If your IP belongs to Tor, DigitalOcean, AWS, or any non-residential ISP, it drastically increases your "Bot Score." 
* **TCP Fingerprinting (p0f):** The WAF analyzes the raw TCP SYN packets. The Window Size, Time to Live (TTL), and TCP Options order are different for Linux/Python network stacks than they are for a standard Windows Chrome browser. If the TCP fingerprint doesn't match the `User-Agent` string you send later, you are flagged.

### Layer 2: TLS Fingerprinting (JA3 / JA4)
If you are using HTTPS, your client must perform a TLS Handshake. This is usually where Python scripts instantly fail.
* **Cipher Suites & Extensions:** When Python `requests` (backed by OpenSSL/urllib3) initiates a TLS connection, it sends a `ClientHello` packet containing supported Cipher Suites, Elliptic Curves, and TLS Extensions in a very specific order. 
* **The Mismatch:** A modern Chrome browser sends a vastly different, highly complex `ClientHello` (often including GREASE extensions to test forward compatibility). Cloudflare calculates an MD5 hash of your `ClientHello` parameters (known as a JA3 fingerprint). If your JA3 hash maps to a known Python library, you receive a `403 Forbidden` before the server even looks at your HTTP headers. 

### Layer 3: HTTP/2 Profiling
If you get past TLS (e.g., by using a modified library like `curl-impersonate`), the WAF analyzes your HTTP/2 frames.
* **Frame Ordering:** Browsers send HTTP/2 `SETTINGS`, `WINDOW_UPDATE`, and `PRIORITY` frames in unique sequences. 
* **Header Ordering (Akamai Fingerprinting):** Chrome always sends pseudo-headers (`:method`, `:authority`, `:scheme`, `:path`) in a specific order, followed by headers like `sec-ch-ua` in a specific order. If your Python script orders them alphabetically or misses a subtle `sec-fetch-site` header, it is flagged as anomalous.

### Layer 4: Client-Side Interrogation (JavaScript & DOM)
If your network requests look perfect (or if you are using Playwright/Selenium), the WAF serves a `200 OK` but returns a payload containing obfuscated JavaScript instead of the HTML you want. This JS executes in your browser to profile the environment.
* **Headless Detection:** The script checks for `navigator.webdriver == true`. It checks if `window.chrome` exists. It checks if the `plugins` array is empty. 
* **Canvas Fingerprinting:** The script forces the browser to render a hidden 3D graphic or text using the HTML5 `<canvas>` element. Headless browsers running on servers (without real GPUs) render these graphics slightly differently at the pixel level due to software-rendering drivers (like Mesa or SwiftShader). The WAF hashes the rendered image and flags software-rendered pixels as bots.
* **Event Tracking:** DataDome tracks micro-movements. If the mouse jumps from `(0,0)` to a button instantly, or if the `Trusted` flag on a click event is `false` (which happens when JavaScript simulates a click rather than a physical hardware interrupt), you are blocked.
