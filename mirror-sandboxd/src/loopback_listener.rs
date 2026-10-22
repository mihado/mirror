use std::io;
use std::net::SocketAddr;
use std::time::Duration;

use tokio::net::TcpListener;

pub(crate) async fn loopback_listener(address: SocketAddr) -> io::Result<TcpListener> {
    let mut attempt = 1;
    loop {
        match TcpListener::bind(address).await {
            Ok(listener) => return Ok(listener),
            Err(error) if attempt >= BIND_ATTEMPTS => return Err(error),
            Err(_) => {
                attempt += 1;
                tokio::time::sleep(BIND_RETRY).await;
            }
        }
    }
}

const BIND_ATTEMPTS: u32 = 20;
const BIND_RETRY: Duration = Duration::from_millis(100);
