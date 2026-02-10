# Flutter Installations Steps
```
🔹 Step:- 1
            Windows / macOS / Linux (same idea)
             Go to flutter.dev → Get Started
             Download the Flutter SDK for your OS
             Extract it somewhere safe, for example:
             Windows: C:\flutter

🔹 Step:-2
           Add Flutter to PATH
            Windows
            Search Environment Variables
            Edit Path
            Add:- C:\flutter\bin

🔹 Step:-3
           Verify Flutter Installation
            Run:- flutter doctor
            This checks:
            Flutter SDK
            VS Code setup

🔹 Step:-4
           Install VS Code + Extensions
             Download VS Code
             Open VS Code → Extensions
           Install:-
                    Flutter
                    Dart

🔹 Step:-5
           open Terminal :- flutter create Folder_Name

🔹 Step:-6
           Run Your Flutter App
             open terminal:- flutter run

```
# Flutter Codes

1. ***Emoji_app***
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
1.1 **Widget.test**
```
   import 'package:flutter_test/flutter_test.dart';



import 'package:emoji_app/main.dart';



void main() {

  testWidgets('Counter increments smoke test', (WidgetTester tester) async {

    // Build our app and trigger a frame.

    await tester.pumpWidget(const EmojiApp());



    // Verify that the initial emoji and button are present.

    expect(find.text('😀'), findsOneWidget);

    expect(find.text('Tap Me'), findsOneWidget);

  });

}

```


2. ***Login_app***
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
2.1 **Widget.test**
```
   import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:login_app/main.dart';

void main() {
  testWidgets('Counter increments smoke test', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const MyApp());

    // Verify that our counter starts at 0.
    expect(find.text('0'), findsOneWidget);
    expect(find.text('1'), findsNothing);

    // Tap the '+' icon and trigger a frame.
    await tester.tap(find.byIcon(Icons.add));
    await tester.pump();

    // Verify that our counter has incremented.
    expect(find.text('0'), findsNothing);
    expect(find.text('1'), findsOneWidget);
  });
}

```
3. ***Emailvalidation_app***
```
  import 'package:flutter/material.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,

      home: const LoginPage(),
    );
  }
}

class LoginPage extends StatefulWidget {
  const LoginPage({super.key});

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final TextEditingController emailController = TextEditingController();

  final TextEditingController passwordController = TextEditingController();

  bool isButtonEnabled = false;

  void validateForm() {
    final email = emailController.text;

    final password = passwordController.text;

    setState(() {
      isButtonEnabled = email.contains('@') && password.length >= 6;
    });
  }

  void showToast() {
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Great! You typed a real email! 😎'),

        duration: Duration(seconds: 2),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Email Validation')),

      body: Padding(
        padding: const EdgeInsets.all(16),

        child: Column(
          children: [
            TextField(
              key: const Key('emailField'),

              controller: emailController,

              onChanged: (_) => validateForm(),

              decoration: const InputDecoration(labelText: 'Email'),
            ),

            const SizedBox(height: 16),

            TextField(
              key: const Key('passwordField'),

              controller: passwordController,

              obscureText: true,

              onChanged: (_) => validateForm(),

              decoration: const InputDecoration(labelText: 'Password'),
            ),

            const SizedBox(height: 24),

            ElevatedButton(
              key: const Key('loginButton'),

              onPressed: isButtonEnabled ? showToast : null,

              child: const Text('Login'),
            ),
          ],
        ),
      ),
    );
  }
}

```
3.1 **Widget.test**
```
   import 'package:flutter/material.dart';

import 'package:flutter_test/flutter_test.dart';

import 'package:emailvalidation_app/main.dart';

void main() {
  testWidgets('Login button enables only with valid email and password', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(const MyApp());

    final emailField = find.byKey(const Key('emailField'));

    final passwordField = find.byKey(const Key('passwordField'));

    final loginButton = find.byKey(const Key('loginButton'));

    // Initially button should be disabled

    ElevatedButton button = tester.widget<ElevatedButton>(loginButton);

    expect(button.onPressed, isNull);

    // Enter invalid email and short password

    await tester.enterText(emailField, 'abc');

    await tester.enterText(passwordField, '123');

    await tester.pump();

    button = tester.widget<ElevatedButton>(loginButton);

    expect(button.onPressed, isNull);

    // Enter valid email and password

    await tester.enterText(emailField, 'test@gmail.com');

    await tester.enterText(passwordField, '123456');

    await tester.pump();

    button = tester.widget<ElevatedButton>(loginButton);

    expect(button.onPressed, isNotNull);

    // Tap login button

    await tester.tap(loginButton);

    await tester.pump();

    // Verify SnackBar text

    expect(find.text('Great! You typed a real email! 😎'), findsOneWidget);
  });
}

```
4. ***Theme Color Change***
```
   import 'package:flutter/material.dart';

void main() {
  runApp(const ThemeChangerApp());
}

class ThemeChangerApp extends StatelessWidget {
  const ThemeChangerApp({super.key});
  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: ThemeChangerScreen(),
    );
  }
}

class ThemeChangerScreen extends StatefulWidget {
  const ThemeChangerScreen({super.key});
  @override
  State<ThemeChangerScreen> createState() => _ThemeChangerScreenState();
}

class _ThemeChangerScreenState extends State<ThemeChangerScreen> {
  Color backgroundColor = Colors.white;
  void changeColor(Color color) {
    setState(() {
      backgroundColor = color;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: backgroundColor,
      appBar: AppBar(
        title: const Text("Theme Color Changer"),
        centerTitle: true,
      ),
      body: Center(
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            ElevatedButton(
              style: ElevatedButton.styleFrom(backgroundColor: Colors.blue),
              onPressed: () => changeColor(Colors.blue.shade100),
              child: const Text("Blue"),
            ),
            ElevatedButton(
              style: ElevatedButton.styleFrom(backgroundColor: Colors.orange),
              onPressed: () => changeColor(Colors.orange.shade100),
              child: const Text("Orange"),
            ),
            ElevatedButton(
              style: ElevatedButton.styleFrom(backgroundColor: Colors.green),
              onPressed: () => changeColor(Colors.green.shade100),
              child: const Text("Green"),
            ),
          ],
        ),
      ),
    );
  }
}

```
4.1 **Widget.test**
```
   import 'package:flutter_test/flutter_test.dart';

import 'package:color_change/main.dart';

void main() {
  testWidgets('Theme color changer smoke test', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const ThemeChangerApp());

    // Verify that the title is displayed.
    expect(find.text('Theme Color Changer'), findsOneWidget);

    // Verify that color change buttons are present.
    expect(find.text('Blue'), findsOneWidget);
    expect(find.text('Orange'), findsOneWidget);
  });
}
   
```
5. ***Counter Auto***
```
   import 'dart:async';
import 'package:flutter/material.dart';

void main() {
  runApp(const CounterApp());
}

class CounterApp extends StatelessWidget {
  const CounterApp({super.key});
  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: CounterScreen(),
    );
  }
}

class CounterScreen extends StatefulWidget {
  const CounterScreen({super.key});
  @override
  State<CounterScreen> createState() => _CounterScreenState();
}

class _CounterScreenState extends State<CounterScreen> {
  int counter = 0;
  Timer? timer;
  void startCounter() {
    timer ??= Timer.periodic(const Duration(seconds: 1), (Timer t) {
      setState(() {
        counter++;
      });
    });
  }

  void pauseCounter() {
    timer?.cancel();
    timer = null;
  }

  void resetCounter() {
    pauseCounter();
    setState(() {
      counter = 0;
    });
  }

  @override
  void dispose() {
    timer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Auto Increment Counter"),
        centerTitle: true,
      ),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text(
            counter.toString(),
            style: const TextStyle(fontSize: 80, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 40),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              ElevatedButton(
                onPressed: startCounter,
                child: const Text("Start"),
              ),
              ElevatedButton(
                onPressed: pauseCounter,
                child: const Text("Pause"),
              ),
              ElevatedButton(
                onPressed: resetCounter,
                child: const Text("Reset"),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

```
5.1 **widget.test**
```
   import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:counterauto_app/main.dart';

void main() {
  testWidgets('Counter increments smoke test', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const CounterApp());

    // Verify that our counter starts at 0.
    expect(find.text('0'), findsOneWidget);
    expect(find.text('1'), findsNothing);

    // Tap the start button and trigger a frame.
    await tester.tap(find.byIcon(Icons.play_arrow));
    await tester.pump(const Duration(seconds: 2));

    // Verify that our counter has incremented.
    expect(find.text('0'), findsNothing);
  });
}
   
```
6. ***Loading Screen(Progress Bar)***
```
   import 'dart:async';
import 'package:flutter/material.dart';

void main() {
  runApp(const LoadingApp());
}

class LoadingApp extends StatelessWidget {
  const LoadingApp({super.key});
  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: LoadingScreen(),
    );
  }
}

class LoadingScreen extends StatefulWidget {
  const LoadingScreen({super.key});
  @override
  State<LoadingScreen> createState() => _LoadingScreenState();
}

class _LoadingScreenState extends State<LoadingScreen> {
  double progress = 0.0;
  String message = "";
  Timer? timer;
  void startLoading() {
    setState(() {
      progress = 0.0;
      message = "Loading your awesome content...";
    });
    timer?.cancel();
    timer = Timer.periodic(const Duration(milliseconds: 100), (Timer t) {
      setState(() {
        progress += 0.01;
        if (progress >= 1.0) {
          progress = 1.0;
          message = "Finished!";
          t.cancel();
        }
      });
    });
  }

  @override
  void dispose() {
    timer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Loading Screen"), centerTitle: true),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            LinearProgressIndicator(
              value: progress,
              minHeight: 20,
              color: Colors.blue,
              backgroundColor: Colors.grey.shade300,
            ),
            const SizedBox(height: 20),
            Text(message, style: const TextStyle(fontSize: 18)),
            const SizedBox(height: 40),
            ElevatedButton(
              onPressed: startLoading,
              child: const Text("Start Loading"),
            ),
          ],
        ),
      ),
    );
  }
}

```
6.1 **widget.test**
```
   import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:loading_screen/main.dart';

void main() {
  testWidgets('Counter increments smoke test', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const LoadingApp());

    // Verify that our counter starts at 0.
    expect(find.text('0'), findsOneWidget);
    expect(find.text('1'), findsNothing);

    // Tap the '+' icon and trigger a frame.
    await tester.tap(find.byIcon(Icons.add));
    await tester.pump();

    // Verify that our counter has incremented.
    expect(find.text('0'), findsNothing);
    expect(find.text('1'), findsOneWidget);
  });
}
   
```
7. ***Reminder App***
```
   import 'package:flutter/material.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';

void main() {
  runApp(const ReminderApp());
}

class ReminderApp extends StatelessWidget {
  const ReminderApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Reminder App',
      home: const ReminderScreen(),
    );
  }
}

class ReminderScreen extends StatefulWidget {
  const ReminderScreen({super.key});
  @override
  State<ReminderScreen> createState() => _ReminderScreenState();
}

class _ReminderScreenState extends State<ReminderScreen> {
  final TextEditingController _controller = TextEditingController();
  FlutterLocalNotificationsPlugin notificationsPlugin =
      FlutterLocalNotificationsPlugin();
  @override
  void initState() {
    super.initState();
    initializeNotification();
  }

  void initializeNotification() async {
    final AndroidInitializationSettings androidSettings =
        AndroidInitializationSettings('@mipmap/ic_launcher');
    final InitializationSettings settings = InitializationSettings(
      android: androidSettings,
    );
    await notificationsPlugin.initialize(settings);
  }

  void showNotification(String message) async {
    final AndroidNotificationDetails androidDetails =
        AndroidNotificationDetails(
          'reminder_channel',
          'Reminders',
          importance: Importance.max,
          priority: Priority.high,
        );
    final NotificationDetails notificationDetails = NotificationDetails(
      android: androidDetails,
    );
    await notificationsPlugin.show(0, 'Reminder', message, notificationDetails);
  }

  void setReminder() {
    String message = _controller.text;
    if (message.isEmpty) return;
    Future.delayed(const Duration(seconds: 5), () {
      showNotification(message);
    });
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(const SnackBar(content: Text('Reminder set for 5 seconds')));
    _controller.clear();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Simple Reminder App'),
        backgroundColor: Colors.blue,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            TextField(
              controller: _controller,
              decoration: const InputDecoration(
                labelText: 'Enter reminder message',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: setReminder,
              child: const Text('Set Reminder'),
            ),
          ],
        ),
      ),
    );
  }
}

```
7.1 **Widget.test**
```
   import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:reminder_app/main.dart';

void main() {
  testWidgets('Counter increments smoke test', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const ReminderApp());

    // Verify that our counter starts at 0.
    expect(find.text('0'), findsOneWidget);
    expect(find.text('1'), findsNothing);

    // Tap the '+' icon and trigger a frame.
    await tester.tap(find.byIcon(Icons.add));
    await tester.pump();

    // Verify that our counter has incremented.
    expect(find.text('0'), findsNothing);
    expect(find.text('1'), findsOneWidget);
  });
}
   
```
8.***Prime Number Finder***
```
 import 'package:flutter/material.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';

void main() {
  runApp(const PrimeApp());
}

class PrimeApp extends StatelessWidget {
  const PrimeApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: const PrimeScreen(),
    );
  }
}

class PrimeScreen extends StatefulWidget {
  const PrimeScreen({super.key});
  @override
  State<PrimeScreen> createState() => _PrimeScreenState();
}

class _PrimeScreenState extends State<PrimeScreen> {
  final TextEditingController lowerController = TextEditingController();
  final TextEditingController upperController = TextEditingController();
  List<int> primes = [];
  bool isLoading = false;
  FlutterLocalNotificationsPlugin notificationsPlugin =
      FlutterLocalNotificationsPlugin();
  @override
  void initState() {
    super.initState();
    initializeNotification();
  }

  void initializeNotification() async {
    final AndroidInitializationSettings androidSettings =
        AndroidInitializationSettings('@mipmap/ic_launcher');
    final InitializationSettings settings = InitializationSettings(
      android: androidSettings,
    );
    await notificationsPlugin.initialize(settings);
  }

  Future<void> showNotification() async {
    final AndroidNotificationDetails androidDetails =
        AndroidNotificationDetails(
          'prime_channel',
          'Prime Finder',
          importance: Importance.max,
          priority: Priority.high,
        );
    final NotificationDetails details = NotificationDetails(
      android: androidDetails,
    );
    await notificationsPlugin.show(
      0,
      'Prime Calculation Done',
      'Prime numbers are ready to view',
      details,
    );
  }

  bool isPrime(int n) {
    if (n <= 1) return false;
    for (int i = 2; i <= n ~/ 2; i++) {
      if (n % i == 0) return false;
    }
    return true;
  }

  Future<void> findPrimesAsync(int low, int high) async {
    setState(() {
      isLoading = true;
      primes.clear();
    });
    await Future.delayed(const Duration(seconds: 1)); // simulate async work
    List<int> result = [];
    for (int i = low; i <= high; i++) {
      if (isPrime(i)) {
        result.add(i);
      }
    }
    setState(() {
      primes = result;
      isLoading = false;
    });
    showNotification();
  }

  void startCalculation() {
    int low = int.parse(lowerController.text);
    int high = int.parse(upperController.text);
    findPrimesAsync(low, high);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Prime Number Finder'),
        backgroundColor: Colors.green,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            TextField(
              controller: lowerController,
              keyboardType: TextInputType.number,
              decoration: const InputDecoration(
                labelText: 'Lower Range',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 10),
            TextField(
              controller: upperController,
              keyboardType: TextInputType.number,
              decoration: const InputDecoration(
                labelText: 'Upper Range',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: startCalculation,
              child: const Text('Find Primes'),
            ),
            const SizedBox(height: 20),
            if (isLoading)
              const CircularProgressIndicator()
            else
              Expanded(
                child: ListView.builder(
                  itemCount: primes.length,
                  itemBuilder: (context, index) {
                    return ListTile(title: Text(primes[index].toString()));
                  },
                ),
              ),
          ],
        ),
      ),
    );
  }
}

```
8.1 **Widget.test**
```
   import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:prime_number_finder/main.dart';

void main() {
  testWidgets('Counter increments smoke test', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const PrimeApp());

    // Verify that our counter starts at 0.
    expect(find.text('0'), findsOneWidget);
    expect(find.text('1'), findsNothing);

    // Tap the '+' icon and trigger a frame.
    await tester.tap(find.byIcon(Icons.add));
    await tester.pump();

    // Verify that our counter has incremented.
    expect(find.text('0'), findsNothing);
    expect(find.text('1'), findsOneWidget);
  });
}
   
```
