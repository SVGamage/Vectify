import 'package:flutter/material.dart';
import '../screens/home_screen.dart';

/// Custom page route for smooth transitions between screens
class SmoothPageRoute<T> extends PageRouteBuilder<T> {
  final Widget page;
  final RouteSettings settings;
  final TransitionType transitionType;

  SmoothPageRoute({
    required this.page,
    required this.settings,
    this.transitionType = TransitionType.fade,
  }) : super(
          settings: settings,
          pageBuilder: (
            BuildContext context,
            Animation<double> animation,
            Animation<double> secondaryAnimation,
          ) =>
              page,
          transitionsBuilder: (
            BuildContext context,
            Animation<double> animation,
            Animation<double> secondaryAnimation,
            Widget child,
          ) {
            switch (transitionType) {
              case TransitionType.fade:
                return FadeTransition(
                  opacity: Tween<double>(begin: 0.0, end: 1.0).animate(
                    CurvedAnimation(
                      parent: animation,
                      curve: Curves.easeOut,
                    ),
                  ),
                  child: child,
                );
              case TransitionType.slideUp:
                return SlideTransition(
                  position: Tween<Offset>(
                    begin: const Offset(0.0, 0.2),
                    end: Offset.zero,
                  ).animate(
                    CurvedAnimation(
                      parent: animation,
                      curve: Curves.easeOut,
                    ),
                  ),
                  child: FadeTransition(
                    opacity: Tween<double>(begin: 0.0, end: 1.0).animate(
                      CurvedAnimation(
                        parent: animation,
                        curve: Curves.easeOut,
                      ),
                    ),
                    child: child,
                  ),
                );
              case TransitionType.slideRight:
                return SlideTransition(
                  position: Tween<Offset>(
                    begin: const Offset(-0.2, 0.0),
                    end: Offset.zero,
                  ).animate(
                    CurvedAnimation(
                      parent: animation,
                      curve: Curves.easeOut,
                    ),
                  ),
                  child: FadeTransition(
                    opacity: Tween<double>(begin: 0.0, end: 1.0).animate(
                      CurvedAnimation(
                        parent: animation,
                        curve: Curves.easeOut,
                      ),
                    ),
                    child: child,
                  ),
                );
              case TransitionType.scale:
                return ScaleTransition(
                  scale: Tween<double>(begin: 0.9, end: 1.0).animate(
                    CurvedAnimation(
                      parent: animation,
                      curve: Curves.easeOut,
                    ),
                  ),
                  child: FadeTransition(
                    opacity: Tween<double>(begin: 0.0, end: 1.0).animate(
                      CurvedAnimation(
                        parent: animation,
                        curve: Curves.easeOut,
                      ),
                    ),
                    child: child,
                  ),
                );
              case TransitionType.slideDown:
                return SlideTransition(
                  position: Tween<Offset>(
                    begin: const Offset(0.0, -0.2),
                    end: Offset.zero,
                  ).animate(
                    CurvedAnimation(
                      parent: animation,
                      curve: Curves.easeOut,
                    ),
                  ),
                  child: FadeTransition(
                    opacity: Tween<double>(begin: 0.0, end: 1.0).animate(
                      CurvedAnimation(
                        parent: animation,
                        curve: Curves.easeOut,
                      ),
                    ),
                    child: child,
                  ),
                );
            }
          },
        );
}

/// Types of transitions available
enum TransitionType {
  fade,
  slideUp,
  slideRight,
  slideDown,
  scale,
}

/// Extension for simple navigation with custom transitions
extension NavigatorExtension on BuildContext {
  Future<T?> pushSmoothRoute<T>({
    required Widget page,
    String? routeName,
    TransitionType transitionType = TransitionType.fade,
  }) {
    return Navigator.of(this).push(
      SmoothPageRoute<T>(
        page: page,
        settings: RouteSettings(name: routeName),
        transitionType: transitionType,
      ),
    );
  }

  Future<bool> popWithAnimation({
    TransitionType transitionType = TransitionType.fade,
    dynamic result,
  }) {
    return Navigator.of(this).maybePop(result);
  }

  /// Navigates to the home screen, clearing the navigation stack
  void navigateToHome() {
    // First try to pop until first route
    Navigator.of(this).popUntil((route) => route.isFirst);

    // If we're not at the HomeScreen, push the HomeScreen
    Future.delayed(const Duration(milliseconds: 100), () {
      if (mounted) {
        try {
          // First check if we can safely pop the current route
          if (Navigator.of(this).canPop()) {
            Navigator.of(this).pushReplacement(
              MaterialPageRoute(
                builder: (context) => const HomeScreen(),
                settings: const RouteSettings(name: 'home'),
              ),
            );
          } else {
            // If we're at the root and can't pop, just push a new route
            Navigator.of(this).push(
              MaterialPageRoute(
                builder: (context) => const HomeScreen(),
                settings: const RouteSettings(name: 'home'),
              ),
            );
          }
        } catch (e) {
          // Last resort, try to directly create a new navigator with HomeScreen
          Navigator.of(this).pushAndRemoveUntil(
            MaterialPageRoute(
              builder: (context) => const HomeScreen(),
              settings: const RouteSettings(name: 'home'),
            ),
            (route) => false, // Remove all routes
          );
        }
      }
    });
  }
}
