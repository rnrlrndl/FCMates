const express = require('express');
const app = express();
const port = 3000;  // 서버 포트 번호, 만약 이 포트를 사용하고 있다면 다른 포트로 사용할것

// 클라이언트에서 HTTP요청 메소드 중 GET를 이용해서 'host:port'로 요청을 보내면 실행되는 라우트
app.get('/',(req,res) => {
    res.sendFile(__dirname + '/views/index.html');
});

app.get('/about', (req,res) => {
    res.sendFile(__dirname + '/views/about.html');
})

app.get('/product', (req,res) => {
    res.sendFile(__dirname + '/views/product.html');
})

// app.listen() 함수를 사용해서 서버를 실행해 준다.
// 클라이언트는 'host:port' 로 노드 서버에 요청을 보낼 수 있다.

app.listen(port, () => {
    console.log('서버가 실행됩니다. http://localhost:${port}');
});

