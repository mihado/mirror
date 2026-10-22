use std::ffi::OsStr;
use std::io::{self, Write};
use std::os::fd::OwnedFd;
use std::path::{Component, Path};

use mirror_sandbox::GuestFile;
use ring::rand::{SecureRandom, SystemRandom};
use rustix::fs::{
    fchmod, mkdirat, open, openat, renameat, statat, unlinkat, AtFlags, FileType, Mode, OFlags,
};
use rustix::io::Errno;

const DIRECTORY: OFlags = OFlags::RDONLY
    .union(OFlags::DIRECTORY)
    .union(OFlags::NOFOLLOW)
    .union(OFlags::CLOEXEC);
const TEMPORARY: OFlags = OFlags::WRONLY
    .union(OFlags::CREATE)
    .union(OFlags::EXCL)
    .union(OFlags::NOFOLLOW)
    .union(OFlags::CLOEXEC);

pub(crate) fn write_file(file: &GuestFile) -> io::Result<()> {
    let path = Path::new(&file.path);
    let name = path
        .file_name()
        .ok_or_else(|| io::Error::other("the path names no file"))?;
    let mut directory = open("/", DIRECTORY, Mode::empty())?;
    for component in path.parent().into_iter().flat_map(Path::components) {
        match component {
            Component::RootDir => {}
            Component::Normal(child) => directory = open_directory(&directory, child)?,
            _ => return Err(io::Error::other("the path is not normalized")),
        }
    }
    refuse_symlink(&directory, name)?;
    let temporary = temporary_name()?;
    let created = openat(
        &directory,
        &temporary,
        TEMPORARY,
        Mode::from_raw_mode(0o600),
    )?;
    let published = fill(created, file)
        .and_then(|()| renameat(&directory, &temporary, &directory, name).map_err(io::Error::from));
    if published.is_err() {
        let _ = unlinkat(&directory, &temporary, AtFlags::empty());
    }
    published
}

fn open_directory(parent: &OwnedFd, name: &OsStr) -> io::Result<OwnedFd> {
    refuse_symlink(parent, name)?;
    match openat(parent, name, DIRECTORY, Mode::empty()) {
        Err(Errno::NOENT) => {}
        opened => return Ok(opened?),
    }
    match mkdirat(parent, name, Mode::from_raw_mode(0o777)) {
        Ok(()) | Err(Errno::EXIST) => {}
        Err(error) => return Err(error.into()),
    }
    Ok(openat(parent, name, DIRECTORY, Mode::empty())?)
}

fn refuse_symlink(directory: &OwnedFd, name: &OsStr) -> io::Result<()> {
    match statat(directory, name, AtFlags::SYMLINK_NOFOLLOW) {
        Ok(stat) if FileType::from_raw_mode(stat.st_mode).is_symlink() => Err(io::Error::other(
            format!("{} is a symbolic link", Path::new(name).display()),
        )),
        _ => Ok(()),
    }
}

fn temporary_name() -> io::Result<String> {
    let mut suffix = [0; 8];
    SystemRandom::new()
        .fill(&mut suffix)
        .map_err(|_| io::Error::other("no randomness for a temporary file name"))?;
    Ok(format!(".mirror-guest-{:016x}", u64::from_le_bytes(suffix)))
}

fn fill(created: OwnedFd, file: &GuestFile) -> io::Result<()> {
    let mut created = std::fs::File::from(created);
    created.write_all(&file.contents)?;
    fchmod(&created, Mode::from_raw_mode(file.mode))?;
    Ok(())
}
