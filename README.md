# Flutter Codes

1. **Emoji_app**
```
import 'dart:math';
import 'package:flutter/material.dart';

void main() {
  runApp(const EmojiApp());
}

class EmojiApp extends StatelessWidget {
  const EmojiApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(debugShowCheckedModeBanner: false, home: EmojiHome());
  }
}

class EmojiHome extends StatefulWidget {
  const EmojiHome({super.key});
  @override
  State<EmojiHome> createState() => _EmojiHomeState();
}

class _EmojiHomeState extends State<EmojiHome> {
  final List<String> emojis = [
    "😀",
    "😁",
    "😂",
    "🤣",
    "😎",
    "😍",
    "🤩",
    "😡",
    "😭",
    "🤔",
    "😴",
    "😇",
  ];

  String currentEmoji = "😀";

  void changeEmoji() {
    final random = Random();
    setState(() {
      currentEmoji = emojis[random.nextInt(emojis.length)];
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(title: const Text("Emoji App"), centerTitle: true),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(currentEmoji, style: const TextStyle(fontSize: 120)),
            const SizedBox(height: 30),
            ElevatedButton(onPressed: changeEmoji, child: const Text("Tap Me")),
          ],
        ),
      ),
    );
  }
}

```

2. **Login_app**
```
 import 'package:flutter/material.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: LoginScreen(),
    );
  }
}

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final TextEditingController userController = TextEditingController();
  final TextEditingController passController = TextEditingController();

  void login() {
    if (userController.text.isNotEmpty && passController.text.isNotEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text("Welcome!"),
          backgroundColor: Colors.green,
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [Colors.blue, Colors.purple],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
        child: Center(
          child: Card(
            elevation: 10,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(15),
            ),
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Text(
                    "Login",
                    style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 20),
                  TextField(
                    key: const Key("username"),
                    controller: userController,
                    decoration: const InputDecoration(
                      labelText: "Username",
                      border: OutlineInputBorder(),
                    ),
                  ),
                  const SizedBox(height: 15),
                  TextField(
                    key: const Key("password"),
                    controller: passController,
                    obscureText: true,
                    decoration: const InputDecoration(
                      labelText: "Password",
                      border: OutlineInputBorder(),
                    ),
                  ),
                  const SizedBox(height: 20),
                  ElevatedButton(
                    key: const Key("loginBtn"),
                    onPressed: login,
                    child: const Text("Login"),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

```
