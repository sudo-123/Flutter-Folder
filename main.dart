

// Import dart math library for Random() class
import 'dart:math';

// Import Flutter Material UI library
import 'package:flutter/material.dart';

// Main function: Entry point of the app
void main() {
  runApp(const EmojiApp()); // Runs the EmojiApp widget
}

// Root widget of the application (Stateless because it doesn't change)
class EmojiApp extends StatelessWidget {
  const EmojiApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false, // Removes debug banner
      home: EmojiHome(), // First screen of the app
    );
  }
}

// A StatefulWidget because the emoji will change on button click
class EmojiHome extends StatefulWidget {
  const EmojiHome({super.key});
  @override
  State<EmojiHome> createState() => _EmojiHomeState();
}

// State class where the UI updates happen
class _EmojiHomeState extends State<EmojiHome> {

  // List of emojis to choose from
  final List<String> emojis = [
    "😀", "😁", "😂", "🤣", "😎", "😍", "🤩", "😡", "😭", "🤔", "😴", "😇"
  ];

  // Default emoji shown at startup
  String currentEmoji = "😀";

  // Function to change emoji randomly
  void changeEmoji() {
    final random = Random(); // Creates a random number generator
    setState(() {
      // Change emoji using a random index
      currentEmoji = emojis[random.nextInt(emojis.length)];
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white, // Screen background color

      // App bar at top
      appBar: AppBar(
        title: const Text("Emoji App"), // Title text
        centerTitle: true, // Center the title
      ),

      // Body of the app
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center, // Center vertically
          children: [

            // Display the current emoji
            Text(
              currentEmoji, // Show the emoji string
              style: const TextStyle(fontSize: 120), // Big emoji size
            ),

            // Space between emoji and button
            const SizedBox(height: 30),

            // Button to change emoji
            ElevatedButton(
              onPressed: changeEmoji, // Call changeEmoji() on click
              child: const Text("Tap Me"), // Button text
            ),
          ],
        ),
      ),
    );
  }
}

