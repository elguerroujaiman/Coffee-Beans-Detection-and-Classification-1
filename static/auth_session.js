(function () {
    var REMEMBER_HINT_COOKIE = "bean_guard_remember_hint";
    var TAB_BOOT_COOKIE = "bean_guard_tab_boot";
    var TAB_MARKER_KEY = "bean_guard_tab_active";

    function hasCookie(name) {
        var key = name + "=";
        return document.cookie.split(";").some(function (cookie) {
            return cookie.trim().indexOf(key) === 0;
        });
    }

    function deleteCookie(name) {
        document.cookie = name + "=; Max-Age=0; Path=/; SameSite=Lax";
    }

    // Persistent login path (remember me checked): allow across tabs/restarts.
    if (hasCookie(REMEMBER_HINT_COOKIE)) {
        try {
            sessionStorage.setItem(TAB_MARKER_KEY, "1");
        } catch (e) {
            // Ignore storage errors; auth remains controlled by backend cookies.
        }
        return;
    }

    var hasTabMarker = false;
    try {
        hasTabMarker = sessionStorage.getItem(TAB_MARKER_KEY) === "1";
    } catch (e) {
        hasTabMarker = false;
    }

    if (hasTabMarker) {
        return;
    }

    // Fresh non-remember login in this tab only.
    if (hasCookie(TAB_BOOT_COOKIE)) {
        try {
            sessionStorage.setItem(TAB_MARKER_KEY, "1");
        } catch (e) {
            // No-op; backend still enforces auth.
        }
        deleteCookie(TAB_BOOT_COOKIE);
        return;
    }

    // New tab/open without remember-me should require login again.
    window.location.replace("/logout");
})();
