---
layout: post
title: "How secure are we?"
date: 2017-09-15
categories: [Technical Fridays]
---

Most of us are connected to the internet in some way or the other and are constantly sharing our information online. But how secure are we in an online world?

> It’s a trick question to ask, we’re secure enough or we’re not secure enough. A lot of it comes down to ‘What does it mean to be secure?’ Everyone’s perception of what secure means and what you should do to be secure is completely different depending on your viewpoint. A good answer to the question ‘Are we secure enough?’ can be: We’re taking reasonable steps. We feel like we’re in a good position, but there’s obviously more we can do
> &mdash; <cite>Andy Ellis at 2015 RSA Conference in San Francisco</cite>

One way security is ensured while you browse the internet is through HTTPS.

HTTPS (Hypertext Transfer Protocol Secure) is the protocol for secure communication on the World Wide Web. HTTPS secures its connections by using Transport Layer Security (TLS) or its predecessor, Secure Sockets Layer (SSL); both frequently referred to as **SSL**. These protocols authenticate web servers and encrypt messages sent between browsers and web servers. HTTPS takes HTTP protocol, and simply layers a SSL/TLS encryption layer on top of it.

SSL certificates creates trust by ensuring a secure connection, shown by  browsers by giving visual clues, such as a lock icon or a green bar.

<img src="/img/SSL.png" style="float: center; display: block; margin: auto; width: 70%; max-width: 100%;">

When a browser attempts to access a website that is secured by SSL, the browser and the web server establishes an SSL connection i.e. SSL handshake. SSL certificates have a key pair: a public and a private key<sup id="a1">[1](#myfootnote1)</sup>. Encryption is done with the public key while decryption using the private key. Because encryption and decryption takes a lot of processing power (computationally costly<sup id="a2">[2](#myfootnote2)</sup>) as assymetric keys being larger than the symmetric keys<sup id="a3">[3](#myfootnote3)</sup>, they are only used during the SSL handshake to create (share) a symmetric session key. After the secure connection is made, the session key is used to encrypt all transmitted data<sup id="a4">[4](#myfootnote4)</sup>.

The steps can be summarised as follows:
1. First, the server (SSL server) sends a copy of its asymmetric public key to the browser (SSL client).
2. The browser creates a symmetric session key and encrypts it with the server's asymmetric public key.        Then sends it to the server.
3. The server decrypts the encrypted session key using its private key.
   The server and the browser now encrypt and decrypt all transmitted data with the symmetric session key. The session key is only used for that session.

<img src="/img/summarySSL.PNG" style="float: center; display: block; margin: auto; width: 75%; max-width: 100%;">

The most important part of an SSL certificate is that it is digitally signed by a trusted party, like DigiCert. Anyone can create a certificate, but browsers only trust certificates that come from an organization on their list of trusted CAs (<abbr title="entity that issues digital certificates, which certifies the ownership of a public key by the named subject of the certificate">Certificate Authority</abbr>).


Note that HTTPS is not unbreakable, and the SSL protocol has to evolve constantly as new attacks against it are discovered and squashed. But it is still an impressively robust way of transmitting data securely without worrying about who sees your messages.



**Footnotes:**  
<a name="myfootnote1"></a>1: [Assymetric Cryptography](https://en.wikipedia.org/wiki/Public-key_cryptography): Assymetrical (public-key) crytography uses pair of keys: *public key* which is publicly available and *private key* which is known to the owner only. Any person can encrypt a message using the public key of the receiver, but such a message can be decrypted only with the receiver's private key. [↩](#a1)  
<a name="myfootnote2"></a>2: [Slow Public Key Algorithms](https://crypto.stackexchange.com/questions/586/why-is-public-key-encryption-so-much-less-efficient-than-secret-key-encryption): The public key and private keys are mathematically related. The public key is available publicly. To compensate, both public and private keys are needed to be quite large to ensure a stronger level of encryption therefore slowing down the encryption process. Also, a third party (<abbr title="Certification Authority">CA</abbr>) has to be introduced to certify the authenticity of a public key. All the above factors contribute to slowness of public key system. [↩](#a2)  
<a name="myfootnote3"></a>3: [Symmetric Cryptography](https://en.wikipedia.org/wiki/Symmetric-key_algorithm): Symmetric (secret-key) cryptography uses a single key for both encryption and decryption of the message. [↩](#a3)  
<a name="myfootnote4"></a>4: [Hybrid Cryptosystem](https://en.wikipedia.org/wiki/Hybrid_cryptosystem): In such cryptosystems, a <abbr title="session key">shared secret key</abbr> is generated by one party, and this much briefer session key is then encrypted by recipient's public key. The recipient then uses his own private key to decrypt the session key. Once both parties have the session key, they can use a much faster symmetric algorithm to encrypt and decrypt messages. [↩](#a4)  
