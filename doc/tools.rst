Tools
=====

Android
-------

``tools/android/install_base.sh`` script installs Android command line tools
for Linux and creates Android Virtual Devices (AVD).

The script creates ``android-sdk-linux`` directory under ``tools/android`` and
sets it as ``ANDROID_HOME`` directory (see
https://developer.android.com/tools/variables).

Your ``ANDROID_USER_HOME`` and ``ANDROID_EMULATOR_HOME`` environment variables
point to ``tools/android/android-sdk-linux/.android``. Hence, removing
``android-sdk-linux`` folder will clean all artefacts of ``install_base.sh``.

It fetches Android command line tools, then installs Android SDK
Platform-Tools, SDK Platform 31 (for Android 12) & 34 (for Android 14), and
Google APIs for platforms 31 & 34 for the associated ABI type.

Finally the script creates AVDs per Pixel 6 for Android 12 & 14.

Shell commands below illustrate how to list available AVDs and run them via
Android emulator:

.. code:: shell

   ANDROID_HOME="/devlib/tools/android/android-sdk-linux"
   export ANDROID_HOME
   EMULATOR="${ANDROID_HOME}/emulator/emulator"

   export ANDROID_EMULATOR_HOME="${ANDROID_HOME}/.android"

   # List available AVDs:
   ${EMULATOR} -list-avds

   # Run devlib-p6-14 AVD in emulator:
   ${EMULATOR} -avd devlib-p6-14 -no-window -no-snapshot -memory 2048 &

   # After ~30 seconds, the emulated device will be ready:
   adb -s emulator-5554 shell "lsmod"

