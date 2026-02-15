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

    if (hasCookie(REMEMBER_HINT_COOKIE)) {
        try {
            sessionStorage.setItem(TAB_MARKER_KEY, "1");
        } catch (e) {
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

    if (hasCookie(TAB_BOOT_COOKIE)) {
        try {
            sessionStorage.setItem(TAB_MARKER_KEY, "1");
        } catch (e) {
        }
        deleteCookie(TAB_BOOT_COOKIE);
        return;
    }

    window.location.replace("/logout");
})();
