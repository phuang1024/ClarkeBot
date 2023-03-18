class File {
    constructor() {
        this.content = "";
    }
}


class Directory {
    constructor() {
        this.children = {};
    }

    add(name, node) {
        this.children[name] = node;
    }

    get(name) {
        return this.children[name];
    }

    remove(name) {
        delete this.children[name];
    }
}


function get_default_fs() {
    let root = new Directory();
    return root;
}
